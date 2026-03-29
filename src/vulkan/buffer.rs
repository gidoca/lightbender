use anyhow::{Context, Result};
use ash::vk;

/// A Vulkan buffer with its backing device memory.
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size:   vk::DeviceSize,
}

impl GpuBuffer {
    pub unsafe fn new(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        mem_props: vk::MemoryPropertyFlags,
    ) -> Result<Self> {
        let buffer = device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .context("create buffer")?;

        let req = device.get_buffer_memory_requirements(buffer);
        let memory_type = find_memory_type(instance, physical_device, req.memory_type_bits, mem_props)
            .context("find memory type")?;

        let memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(memory_type),
                None,
            )
            .context("allocate memory")?;

        device.bind_buffer_memory(buffer, memory, 0)?;

        Ok(Self { buffer, memory, size })
    }

    /// Write `data` into a HOST_VISIBLE buffer via mapped memory.
    pub unsafe fn upload_slice<T: bytemuck::Pod>(
        &self,
        device: &ash::Device,
        data: &[T],
    ) -> Result<()> {
        let bytes = bytemuck::cast_slice(data);
        assert!(bytes.len() as vk::DeviceSize <= self.size, "upload_slice: data ({} bytes) exceeds buffer size ({})", bytes.len(), self.size);
        let ptr = device
            .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
            .context("map memory")?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
        device.unmap_memory(self.memory);
        Ok(())
    }

    pub unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_buffer(self.buffer, None);
        device.free_memory(self.memory, None);
    }
}

/// Upload `data` to a DEVICE_LOCAL buffer via a staging buffer.
pub unsafe fn upload_to_device_local<T: bytemuck::Pod>(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    usage: vk::BufferUsageFlags,
    data: &[T],
) -> Result<GpuBuffer> {
    let bytes = bytemuck::cast_slice::<T, u8>(data);
    let size = bytes.len() as vk::DeviceSize;

    // Staging buffer (host-visible)
    let staging = GpuBuffer::new(
        device,
        instance,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    staging.upload_slice(device, data)?;

    // Device-local destination
    let dst = GpuBuffer::new(
        device,
        instance,
        physical_device,
        size,
        usage | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // One-shot copy
    let cmd = begin_one_shot(device, command_pool)?;
    device.cmd_copy_buffer(
        cmd,
        staging.buffer,
        dst.buffer,
        &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }],
    );
    end_one_shot(device, command_pool, queue, cmd)?;

    staging.destroy(device);
    Ok(dst)
}

pub unsafe fn begin_one_shot(
    device: &ash::Device,
    pool: vk::CommandPool,
) -> Result<vk::CommandBuffer> {
    let cmd = device
        .allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
        .context("alloc one-shot cmd")?[0];
    device.begin_command_buffer(
        cmd,
        &vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
    )?;
    Ok(cmd)
}

pub unsafe fn end_one_shot(
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    device.end_command_buffer(cmd)?;
    let cmds = [cmd];
    let submit = vk::SubmitInfo::default().command_buffers(&cmds);
    device.queue_submit(queue, &[submit], vk::Fence::null())?;
    device.queue_wait_idle(queue)?;
    device.free_command_buffers(pool, &cmds);
    Ok(())
}

pub fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    props: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    for i in 0..mem_props.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && mem_props.memory_types[i as usize].property_flags.contains(props)
        {
            return Ok(i);
        }
    }
    anyhow::bail!("no suitable memory type for flags {props:?}")
}

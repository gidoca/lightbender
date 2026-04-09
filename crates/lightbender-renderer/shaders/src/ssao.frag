#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out float outAO;

// G-buffer inputs
layout(set = 0, binding = 0) uniform sampler2D gDepth;
layout(set = 0, binding = 1) uniform sampler2D gNormal;
layout(set = 0, binding = 2) uniform sampler2D noiseTex;

// SSAO kernel + parameters
layout(set = 0, binding = 3) uniform SsaoParams {
    vec4  samples[64];
    mat4  projection;
    mat4  inverseProjection;
    float radius;
    float bias;
    float power;
    float _pad;
    vec2  noiseScale; // viewport / noiseTexSize
    vec2  _pad2;
} ssao;

const int KERNEL_SIZE = 64;

vec3 viewPosFromDepth(vec2 uv, float depth) {
    // NDC: x,y in [-1,1], z in [0,1] (Vulkan)
    vec4 clipPos = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 viewPos = ssao.inverseProjection * clipPos;
    return viewPos.xyz / viewPos.w;
}

void main() {
    float depth = texture(gDepth, inUV).r;
    if (depth >= 1.0) {
        outAO = 1.0;
        return;
    }

    vec3 fragPos = viewPosFromDepth(inUV, depth);
    // Decode normal from [0,1] to [-1,1]
    vec3 normal  = normalize(texture(gNormal, inUV).rgb * 2.0 - 1.0);

    // Random rotation vector from tiled noise texture
    vec3 randomVec = vec3(texture(noiseTex, inUV * ssao.noiseScale).rg, 0.0);

    // Gram-Schmidt to build TBN from normal + randomVec
    vec3 tangent   = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN       = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        // Sample position in view space
        vec3 samplePos = fragPos + TBN * ssao.samples[i].xyz * ssao.radius;

        // Project sample to screen
        vec4 offset = ssao.projection * vec4(samplePos, 1.0);
        offset.xy  /= offset.w;
        offset.xy   = offset.xy * 0.5 + 0.5;

        // Sample depth at projected position
        float sampleDepth = texture(gDepth, offset.xy).r;
        vec3  sampleViewPos = viewPosFromDepth(offset.xy, sampleDepth);
        float actualDepth = sampleViewPos.z;

        // Range check: only occlude within radius
        float rangeCheck = smoothstep(0.0, 1.0,
            ssao.radius / abs(fragPos.z - actualDepth));
        occlusion += (actualDepth >= samplePos.z + ssao.bias ? 1.0 : 0.0) * rangeCheck;
    }

    occlusion = 1.0 - (occlusion / float(KERNEL_SIZE));
    outAO = pow(occlusion, ssao.power);
}

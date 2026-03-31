#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

struct GpuLight {
    vec4  positionOrDirection;
    vec3  color;
    float intensity;
    float range;
    float _pad0;
    vec2  spotAngles;
    vec4  _pad1;
};

layout(set = 0, binding = 0) uniform FrameUniforms {
    mat4     view;
    mat4     projection;
    vec4     cameraPosition;
    GpuLight lights[8];
    uint     lightCount;
    float    envIntensity;
} frame;

layout(set = 2, binding = 0) uniform sampler2D envMap;

const float PI = 3.14159265359;

vec2 directionToEquirect(vec3 dir) {
    float phi   = atan(dir.z, dir.x);                  // [-PI, PI]
    float theta = asin(clamp(dir.y, -1.0, 1.0));       // [-PI/2, PI/2]
    return vec2(phi / (2.0 * PI) + 0.5, -theta / PI + 0.5);
}

void main() {
    // Reconstruct clip-space position
    vec2 ndc = inUV * 2.0 - 1.0;

    // Inverse view-projection to get world-space direction
    mat4 invVP = inverse(frame.projection * frame.view);
    vec4 worldPos = invVP * vec4(ndc, 1.0, 1.0);
    vec3 dir = normalize(worldPos.xyz / worldPos.w - frame.cameraPosition.xyz);

    vec2 uv = directionToEquirect(dir);
    vec3 color = texture(envMap, uv).rgb * frame.envIntensity;

    // Reinhard tone mapping + gamma correction
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, 1.0);
}

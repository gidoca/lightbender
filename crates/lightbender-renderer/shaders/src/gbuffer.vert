#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;

layout(set = 0, binding = 0) uniform FrameUniforms {
    mat4 view;
    mat4 projection;
    vec4 cameraPosition;
} frame;

layout(push_constant) uniform PushConstants {
    mat4 model;
} push;

layout(location = 0) out vec3 vViewNormal;

void main() {
    mat3 normalMatrix = transpose(inverse(mat3(push.model)));
    vec3 worldNormal  = normalize(normalMatrix * inNormal);
    vViewNormal       = normalize(mat3(frame.view) * worldNormal);

    vec4 worldPos = push.model * vec4(inPosition, 1.0);
    gl_Position   = frame.projection * frame.view * worldPos;
}

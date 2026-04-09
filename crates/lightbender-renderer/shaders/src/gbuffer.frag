#version 450

layout(location = 0) in vec3 vViewNormal;

layout(location = 0) out vec4 outNormal;

void main() {
    vec3 N = normalize(vViewNormal);
    // Encode from [-1,1] to [0,1] for UNORM storage
    outNormal = vec4(N * 0.5 + 0.5, 1.0);
}

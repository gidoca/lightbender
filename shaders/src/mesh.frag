#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragWorldPos;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform FrameUniforms {
    mat4 view;
    mat4 projection;
    vec4 cameraPosition;
} frame;

void main() {
    vec3 N = normalize(fragNormal);
    // Simple directional light from above-right
    vec3 L = normalize(vec3(1.0, 2.0, 1.5));
    float diff = max(dot(N, L), 0.0);
    vec3 color = vec3(0.8, 0.6, 0.3); // warm orange base color
    vec3 ambient = color * 0.15;
    vec3 diffuse = color * diff;

    // Specular (Blinn-Phong)
    vec3 V = normalize(frame.cameraPosition.xyz - fragWorldPos);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 64.0);
    vec3 specular = vec3(0.4) * spec;

    outColor = vec4(ambient + diffuse + specular, 1.0);
}

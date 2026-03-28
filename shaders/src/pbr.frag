#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in mat3 fragTBN;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform FrameUniforms {
    mat4 view;
    mat4 projection;
    vec4 cameraPosition;
} frame;

// Material textures (set 1)
layout(set = 1, binding = 0) uniform sampler2D texBaseColor;
layout(set = 1, binding = 1) uniform sampler2D texNormal;
layout(set = 1, binding = 2) uniform sampler2D texMetallicRoughness;
layout(set = 1, binding = 3) uniform sampler2D texOcclusion;
layout(set = 1, binding = 4) uniform sampler2D texEmissive;

const float PI = 3.14159265359;

// ── PBR helper functions ──────────────────────────────────────────────────────

float distributionGGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(float NdotV, float NdotL, float roughness) {
    return geometrySchlickGGX(NdotV, roughness)
         * geometrySchlickGGX(NdotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ── Main ──────────────────────────────────────────────────────────────────────

void main() {
    // Sample textures
    vec4 baseColorSample    = texture(texBaseColor, fragUV);
    vec3 normalSample       = texture(texNormal, fragUV).rgb;
    vec2 metallicRoughness  = texture(texMetallicRoughness, fragUV).bg; // B=metallic, G=roughness
    float occlusion         = texture(texOcclusion, fragUV).r;
    vec3 emissive           = texture(texEmissive, fragUV).rgb;

    // Alpha cutout
    if (baseColorSample.a < 0.01) discard;

    vec3 albedo    = pow(baseColorSample.rgb, vec3(2.2)); // sRGB → linear
    float metallic  = metallicRoughness.x;
    float roughness = max(metallicRoughness.y, 0.04);

    // Normal mapping
    vec3 N = normalSample * 2.0 - 1.0;
    N = normalize(fragTBN * N);

    vec3 V = normalize(frame.cameraPosition.xyz - fragWorldPos);

    // Reflectance at normal incidence (F0)
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Hardcoded directional light (sun) — will be replaced by UBO lights later
    vec3 lightDir   = normalize(vec3(1.0, 2.0, 1.5));
    vec3 lightColor = vec3(3.0, 2.85, 2.55);

    // Radiance contribution from one directional light
    vec3 Lo = vec3(0.0);
    {
        vec3 L = lightDir;
        vec3 H = normalize(V + L);

        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);

        float D = distributionGGX(NdotH, roughness);
        float G = geometrySmith(NdotV, NdotL, roughness);
        vec3  F = fresnelSchlick(HdotV, F0);

        vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        Lo += (kD * albedo / PI + specular) * lightColor * NdotL;
    }

    // Ambient (IBL approximation)
    vec3 ambient = vec3(0.03) * albedo * occlusion;

    vec3 color = ambient + Lo + emissive * 3.0;

    // Reinhard tone mapping + gamma correction
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, baseColorSample.a);
}

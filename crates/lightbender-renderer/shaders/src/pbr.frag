#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in mat3 fragTBN;

layout(location = 0) out vec4 outColor;

struct GpuLight {
    vec4  positionOrDirection;
    vec3  color;
    float intensity;
    float range;
    float _pad0;
    vec2  spotAngles;
    int   shadowVPIndex;
    float shadowBias;
    vec2  _pad1;
};

layout(set = 0, binding = 0) uniform FrameUniforms {
    mat4     view;
    mat4     projection;
    vec4     cameraPosition;
    GpuLight lights[8];
    uint     lightCount;
    float    envIntensity;
    uint     shadowCount;
    uint     _pad;
    mat4     shadowVP[4];
} frame;

// Material textures (set 1)
layout(set = 1, binding = 0) uniform sampler2D texBaseColor;
layout(set = 1, binding = 1) uniform sampler2D texNormal;
layout(set = 1, binding = 2) uniform sampler2D texMetallicRoughness;
layout(set = 1, binding = 3) uniform sampler2D texOcclusion;
layout(set = 1, binding = 4) uniform sampler2D texEmissive;

// Environment map (set 2)
layout(set = 2, binding = 0) uniform sampler2D envMap;

// Shadow map array (set 3)
layout(set = 3, binding = 0) uniform sampler2DArray shadowMap;

// Material factors (push constants, offset 64)
layout(push_constant) uniform MaterialFactors {
    layout(offset = 64) vec4  baseColorFactor;
    layout(offset = 80) vec3  emissiveFactor;
    layout(offset = 92) float metallicFactor;
    layout(offset = 96) float roughnessFactor;
} material;

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

vec2 directionToEquirect(vec3 dir) {
    float phi   = atan(dir.z, dir.x);
    float theta = asin(clamp(dir.y, -1.0, 1.0));
    return vec2(phi / (2.0 * PI) + 0.5, -theta / PI + 0.5);
}

float sampleShadow(uint lightIdx, vec3 worldPos) {
    int vpIdx = frame.lights[lightIdx].shadowVPIndex;
    if (vpIdx < 0) return 1.0;

    vec4 lsPos = frame.shadowVP[vpIdx] * vec4(worldPos, 1.0);
    vec3 proj = lsPos.xyz / lsPos.w;
    // Vulkan clip: x,y in [-1,1], z in [0,1]
    vec2 uv = proj.xy * 0.5 + 0.5;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0)
        return 1.0;

    float bias = 0.005;

    // 3×3 PCF with manual depth comparison
    float shadow = 0.0;
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0).xy);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float depth = texture(shadowMap, vec3(uv + vec2(x, y) * texelSize, float(vpIdx))).r;
            shadow += (proj.z - bias > depth) ? 0.0 : 1.0;
        }
    }
    return shadow / 9.0;
}

// ── Main ──────────────────────────────────────────────────────────────────────

void main() {
    // Sample textures
    vec4 baseColorSample    = texture(texBaseColor, fragUV);
    vec3 normalSample       = texture(texNormal, fragUV).rgb;
    vec2 metallicRoughness  = texture(texMetallicRoughness, fragUV).bg; // B=metallic, G=roughness
    float occlusion         = texture(texOcclusion, fragUV).r;
    vec3 emissive           = texture(texEmissive, fragUV).rgb;

    // Apply material factors (both texture samples and factors are in linear space;
    // sRGB→linear conversion is handled by the VK_FORMAT_R8G8B8A8_SRGB texture format)
    baseColorSample *= material.baseColorFactor;
    emissive = emissive * material.emissiveFactor;

    // Alpha cutout
    if (baseColorSample.a < 0.01) discard;

    vec3 albedo    = baseColorSample.rgb;
    float metallic  = metallicRoughness.x * material.metallicFactor;
    float roughness = max(metallicRoughness.y * material.roughnessFactor, 0.04);

    // Normal mapping
    vec3 N = normalSample * 2.0 - 1.0;
    N = normalize(fragTBN * N);

    // Flip normal for back-facing fragments (double-sided rendering)
    if (!gl_FrontFacing) {
        N = -N;
    }

    vec3 V = normalize(frame.cameraPosition.xyz - fragWorldPos);

    // Reflectance at normal incidence (F0)
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Accumulate radiance from all scene lights
    float NdotV = max(dot(N, V), 0.001);
    vec3 Lo = vec3(0.0);

    // Fallback: if no lights are defined and no environment map, use a default sun
    uint numLights = frame.lightCount;
    if (numLights == 0u && frame.envIntensity == 0.0) {
        // Default directional light (warm sunlight)
        vec3 L = normalize(vec3(1.0, 2.0, 1.5));
        vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);
        float D = distributionGGX(NdotH, roughness);
        float G = geometrySmith(NdotV, NdotL, roughness);
        vec3  F = fresnelSchlick(HdotV, F0);
        vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        Lo += (kD * albedo / PI + specular) * vec3(3.0, 2.85, 2.55) * NdotL;
    }

    for (uint i = 0u; i < numLights; i++) {
        GpuLight light = frame.lights[i];
        float lightType = light.positionOrDirection.w;

        vec3 L;
        float attenuation = 1.0;

        if (lightType < 0.5) {
            // Directional light (w=0): direction points toward the light source
            L = normalize(-light.positionOrDirection.xyz);
        } else {
            // Point (w=1) or spot (w=2) light
            vec3 toLight = light.positionOrDirection.xyz - fragWorldPos;
            float dist = length(toLight);
            L = toLight / max(dist, 0.0001);

            // Distance attenuation (inverse-square with range cutoff)
            float rangeAtt = max(1.0 - pow(dist / light.range, 4.0), 0.0);
            attenuation = (rangeAtt * rangeAtt) / max(dist * dist, 0.0001);

            // Spot cone attenuation (w=2)
            if (lightType > 1.5) {
                // spotAngles.x = cos(inner), spotAngles.y = cos(outer)
                // Note: direction for spot needs to be stored; for now use -L as approximation
                // TODO: store spot direction separately when GpuLight is extended
                float theta = dot(-L, normalize(-light.positionOrDirection.xyz));
                float epsilon = light.spotAngles.x - light.spotAngles.y;
                float spotAtt = clamp((theta - light.spotAngles.y) / max(epsilon, 0.0001), 0.0, 1.0);
                attenuation *= spotAtt;
            }
        }

        vec3 radiance = light.color * light.intensity * attenuation;

        vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);

        float D = distributionGGX(NdotH, roughness);
        float G = geometrySmith(NdotV, NdotL, roughness);
        vec3  F = fresnelSchlick(HdotV, F0);

        vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        float shadow = sampleShadow(i, fragWorldPos);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL * shadow;
    }

    // Image-based lighting (IBL) from environment map
    // Clamp samples to prevent extreme HDR values from causing overexposure.
    // Without a pre-convolved irradiance map, point-sampling raw HDR radiance
    // can return extreme values (e.g. sun disc); clamping approximates the
    // hemisphere-averaging effect of a proper irradiance map.
    vec3 rawIrradiance = texture(envMap, directionToEquirect(N)).rgb;
    vec3 irradiance = min(rawIrradiance, vec3(3.0));
    vec3 diffuseIBL = irradiance * (albedo / PI) * (1.0 - metallic);

    vec3 R = reflect(-V, N);
    vec3 rawSpecEnv = texture(envMap, directionToEquirect(R)).rgb;
    // Allow higher values for specular to preserve bright reflections
    vec3 specEnv = min(rawSpecEnv, vec3(20.0));
    vec3 F_ibl = fresnelSchlick(max(dot(N, V), 0.0), F0);
    vec3 specularIBL = specEnv * F_ibl;

    vec3 ambient = (diffuseIBL + specularIBL) * occlusion * frame.envIntensity;

    vec3 color = ambient + Lo + emissive * 3.0;

    // Reinhard tone mapping + gamma correction
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    // Output alpha: use base color alpha, modulated by Fresnel for transparent materials
    float alpha = baseColorSample.a;
    if (alpha < 1.0) {
        // Increase opacity at grazing angles (Fresnel effect for glass-like materials)
        float fresnel = pow(1.0 - max(dot(N, V), 0.0), 5.0);
        alpha = mix(alpha, 1.0, fresnel);
    }
    outColor = vec4(color, alpha);
}

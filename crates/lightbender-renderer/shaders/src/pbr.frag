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
    float lightSizeUV;
    float _pad1;
};

struct GpuAreaLight {
    vec4  p0;
    vec4  p1;
    vec4  p2;
    vec4  p3;
    vec3  color;
    float intensity;
    int   shadowVPIndex;
    float lightSizeUV;
    vec2  _pad;
};

layout(set = 0, binding = 0) uniform FrameUniforms {
    mat4         view;
    mat4         projection;
    vec4         cameraPosition;
    GpuLight     lights[8];
    uint         lightCount;
    float        envIntensity;
    uint         shadowCount;
    uint         areaLightCount;
    mat4         shadowVP[4];
    GpuAreaLight areaLights[4];
    mat4         inverseProjection;
    vec4         ssaoParams; // x=radius, y=bias, z=power, w=enable
    vec4         screenSize; // xy = screen width/height in pixels
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

// LTC LUTs (set 4) — see crates/lightbender-renderer/src/ltc.rs
layout(set = 4, binding = 0) uniform sampler2D ltcMat;
layout(set = 4, binding = 1) uniform sampler2D ltcMag;

// SSAO (set 5) — screen-space ambient occlusion result
layout(set = 5, binding = 0) uniform sampler2D ssaoTex;

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

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec2 directionToEquirect(vec3 dir) {
    float phi   = atan(dir.z, dir.x);
    float theta = asin(clamp(dir.y, -1.0, 1.0));
    return vec2(phi / (2.0 * PI) + 0.5, -theta / PI + 0.5);
}

// 16-tap Poisson disk for blocker search and PCF.
const vec2 POISSON_DISK_16[16] = vec2[](
    vec2(-0.94201624, -0.39906216), vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870), vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845), vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554), vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507), vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367), vec2( 0.14383161, -0.14100790)
);

// PCSS shadow lookup. `lightSizeUV` is the world-space half-extent of the
// emitter projected to UVs at the shadow frustum's near plane; pass 0 to fall
// back to a tight 4-tap PCF.
float sampleShadowPCSS(int vpIdx, vec3 worldPos, float lightSizeUV) {
    if (vpIdx < 0) return 1.0;

    vec4 lsPos = frame.shadowVP[vpIdx] * vec4(worldPos, 1.0);
    vec3 proj = lsPos.xyz / lsPos.w;
    // Vulkan clip: x,y in [-1,1], z in [0,1]
    vec2 uv = proj.xy * 0.5 + 0.5;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0)
        return 1.0;

    float bias = 0.0005;
    float receiver = proj.z;

    // Per-pixel rotation of the Poisson disk so neighbouring fragments don't
    // sample the same shadow-map texels in lockstep (interleaved gradient
    // noise — Jorge Jiménez, "Next Generation Post Processing in Call of Duty").
    float ign = fract(52.9829189 *
        fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float ang = ign * 6.2831853;
    float cs = cos(ang);
    float sn = sin(ang);
    mat2 rot = mat2(cs, -sn, sn, cs);

    // Cheap path for hard/very small lights.
    if (lightSizeUV <= 0.0) {
        vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0).xy);
        float s = 0.0;
        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                float depth = texture(shadowMap, vec3(uv + vec2(x, y) * texelSize, float(vpIdx))).r;
                s += (receiver - bias > depth) ? 0.0 : 1.0;
            }
        }
        return s / 9.0;
    }

    // ── Phase 1: blocker search ────────────────────────────────────────────
    float blockerSum  = 0.0;
    float blockerCnt  = 0.0;
    float searchRadius = lightSizeUV;
    for (int i = 0; i < 16; i++) {
        vec2 offset = rot * POISSON_DISK_16[i] * searchRadius;
        float depth = texture(shadowMap, vec3(uv + offset, float(vpIdx))).r;
        if (depth + bias < receiver) {
            blockerSum += depth;
            blockerCnt += 1.0;
        }
    }
    if (blockerCnt < 0.5) return 1.0;
    float avgBlocker = blockerSum / blockerCnt;

    // ── Phase 2: penumbra estimate ─────────────────────────────────────────
    // Lower-bound the kernel at ~2 texels so that all 16 taps don't collapse
    // into a single shadow-map texel (which produces visible texel-grid
    // banding even with rotation).
    float wPenumbra = max((receiver - avgBlocker) * lightSizeUV / max(avgBlocker, 1e-4), 0.0);
    wPenumbra = clamp(wPenumbra, 2.0 / float(textureSize(shadowMap, 0).x), 0.05);

    // ── Phase 3: variable PCF ──────────────────────────────────────────────
    float shadow = 0.0;
    for (int i = 0; i < 16; i++) {
        vec2 offset = rot * POISSON_DISK_16[i] * wPenumbra;
        float depth = texture(shadowMap, vec3(uv + offset, float(vpIdx))).r;
        shadow += (receiver - bias > depth) ? 0.0 : 1.0;
    }
    return shadow / 16.0;
}

float sampleShadow(uint lightIdx, vec3 worldPos) {
    return sampleShadowPCSS(
        frame.lights[lightIdx].shadowVPIndex,
        worldPos,
        frame.lights[lightIdx].lightSizeUV
    );
}

// ── Linearly Transformed Cosines (Heitz) ──────────────────────────────────────
const float LTC_LUT_SIZE  = 64.0;
const float LTC_LUT_SCALE = (LTC_LUT_SIZE - 1.0) / LTC_LUT_SIZE;
const float LTC_LUT_BIAS  = 0.5 / LTC_LUT_SIZE;

vec2 ltcLutUv(float roughness, float NdotV) {
    vec2 uv = vec2(roughness, sqrt(1.0 - clamp(NdotV, 0.0, 1.0)));
    return uv * LTC_LUT_SCALE + LTC_LUT_BIAS;
}

float ltcIntegrateEdge(vec3 v1, vec3 v2) {
    float cosTheta = clamp(dot(v1, v2), -1.0, 1.0);
    float theta = acos(cosTheta);
    float sinTheta = sin(theta);
    float k = (theta > 1e-4) ? theta / sinTheta : 1.0;
    return cross(v1, v2).z * k;
}

float ltcEvaluate(vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4], bool twoSided) {
    // Build orthonormal basis around N with V in the +x half-plane.
    vec3 T1 = normalize(V - N * dot(V, N));
    vec3 T2 = cross(N, T1);
    Minv = Minv * transpose(mat3(T1, T2, N));

    vec3 L0 = normalize(Minv * (points[0] - P));
    vec3 L1 = normalize(Minv * (points[1] - P));
    vec3 L2 = normalize(Minv * (points[2] - P));
    vec3 L3 = normalize(Minv * (points[3] - P));

    float sum = 0.0;
    sum += ltcIntegrateEdge(L0, L1);
    sum += ltcIntegrateEdge(L1, L2);
    sum += ltcIntegrateEdge(L2, L3);
    sum += ltcIntegrateEdge(L3, L0);

    return twoSided ? abs(sum) : max(0.0, -sum);
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

    // ── Area lights via LTC ───────────────────────────────────────────────────
    if (frame.areaLightCount > 0u) {
        vec2 lutUv = ltcLutUv(roughness, NdotV);
        vec4 t1 = texture(ltcMat, lutUv);
        vec4 t2 = texture(ltcMag, lutUv);
        // GGX inverse-matrix layout: stored as four scalars (a,b,c,d) packed in t1.
        mat3 Minv = mat3(
            vec3(  1.0, 0.0, t1.y),
            vec3(  0.0, t1.z, 0.0),
            vec3( t1.w, 0.0, t1.x)
        );

        for (uint i = 0u; i < frame.areaLightCount; i++) {
            GpuAreaLight a = frame.areaLights[i];
            vec3 pts[4];
            pts[0] = a.p0.xyz;
            pts[1] = a.p1.xyz;
            pts[2] = a.p2.xyz;
            pts[3] = a.p3.xyz;

            // Diffuse: identity transform. One-sided so back-facing emitters
            // (e.g. the Cornell ceiling light viewed from below) don't bleed
            // light onto faces that point away from them.
            float diffuse = ltcEvaluate(N, V, fragWorldPos, mat3(1.0), pts, false);
            // Specular: GGX-fitted matrix.
            float spec = ltcEvaluate(N, V, fragWorldPos, Minv, pts, false);
            // Heitz BRDF approximation: F0 * scale + (1 - F0) * fresnel.
            vec3 specColor = F0 * t2.x + (vec3(1.0) - F0) * t2.y;

            vec3 kD = (vec3(1.0) - specColor) * (1.0 - metallic);
            float aShadow = sampleShadowPCSS(a.shadowVPIndex, fragWorldPos, a.lightSizeUV);

            // 1/(2π) normalisation factor that pulls the analytical edge sum
            // into the range expected by the rest of the BRDF.
            const float INV_2PI = 0.15915494;
            vec3 areaRadiance = a.color * a.intensity;
            Lo += (kD * albedo * diffuse + specColor * spec)
                  * areaRadiance * INV_2PI * aShadow;
        }
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
    // Use roughness-aware Fresnel: reduces specular reflection for rough surfaces
    vec3 F_ibl = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    // Without a pre-filtered environment map, attenuate specular by roughness
    // to approximate the blurring effect of rough surfaces
    vec3 specularIBL = specEnv * F_ibl * (1.0 - roughness * roughness);

    float ssaoFactor = 1.0;
    if (frame.ssaoParams.w > 0.5) {
        ssaoFactor = texture(ssaoTex, gl_FragCoord.xy / frame.screenSize.xy).r;
    }
    vec3 ambient = (diffuseIBL + specularIBL) * occlusion * ssaoFactor * frame.envIntensity;

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

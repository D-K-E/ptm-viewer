# author: Kaan Eraslan
# license: see, LICENSE.
# Simple file for holding shaders

lambertVshader = """
#version 330 core

in vec3 aPos;
in vec3 aNormal;
in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
}
"""

lambertFshader = """
#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D diffuseMap;
uniform vec3 lightColor;
uniform vec3 lightPos;

void main() {
    vec3 surfaceNormal = normalize(Normal);
    vec3 lightDirection = normalize(lightPos - FragPos);
    vec3 diffuseColor = texture(diffuseMap, TexCoord).rgb;
    float costheta = max(dot(lightDirection, surfaceNormal), 0.0);

    // lambertian term: k_d * (N \cdot L) * I_p
    FragColor = vec4(diffuseColor * lightColor * costheta, 1.0);
}
"""

phongVshader = """
#version 330 core

in vec3 aPos;
in vec3 aNormal;
in vec2 aTexCoord;

uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

void main()
{
    // FragPos = vec3(model * vec4(aPos, 1.0));
    FragPos = aPos;
    gl_Position = projection * view * model * vec4(FragPos, 1.0);
    // Normal = mat3(transpose(inverse(model))) * aNormal;
    Normal = aNormal;
    TexCoord = aTexCoord;
}
"""

phongFshader = """
#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

uniform bool blinn;

struct SpotLightSource {

    vec3 position;
    vec3 direction;
    float cutOff;
    float outerCutOff;
    vec3 attenuation;
    vec3 color;

};
uniform SpotLightSource light;
uniform vec3 viewerPosition;
uniform float ambientCoeff;

struct PtmMaterial {
    sampler2D diffuseMap; // object colors per fragment
    sampler2D normalMap1; // normals per vertex
    sampler2D normalMap2; // normals per vertex
    sampler2D normalMap3; // normals per vertex
    float shininess;
};

uniform PtmMaterial material;

float computeColorPerChannelPhong(vec3 normal,
                                  vec3 f_pos,
                                  float objIntensity,
                                  float ambientIntensity,
                                  float lightIntensity,
                                  float shininess,
                                  SpotLightSource light,
                                  bool blinn,
                                  vec3 viewDir);
float computeAttenuation(vec3 att, float distVal);

void main() {
    // obtain normal map from texture [0,1]
    vec3 normalr = texture(material.normalMap1, TexCoord).rgb;
    vec3 normalg = texture(material.normalMap2, TexCoord).rgb;
    vec3 normalb = texture(material.normalMap3, TexCoord).rgb;
    // transform vector to range [-1,1]
    normalr = normalize((normalr * 2.0) - 1.0);
    normalg = normalize((normalg * 2.0) - 1.0);
    normalb = normalize((normalb * 2.0) - 1.0);
    vec3 objColor = texture(material.diffuseMap, TexCoord).rgb;
    vec3 viewDirection = normalize(viewerPosition - FragPos);
    float red = computeColorPerChannelPhong(normalr, FragPos, objColor.x,
                                            ambientCoeff, light.color.x,
                                            material.shininess,
                                            light, blinn, viewDirection);
    float green = computeColorPerChannelPhong(normalg, FragPos, objColor.y,
                                              ambientCoeff, light.color.y,
                                              material.shininess,
                                              light, blinn, viewDirection);
    float blue = computeColorPerChannelPhong(normalg, FragPos, objColor.z,
                                             ambientCoeff, light.color.z,
                                             material.shininess,
                                             light, blinn, viewDirection);
    FragColor = vec4(red, green, blue, 1.0);
}
float computeColorPerChannelPhong(vec3 normal,
                                  vec3 f_pos,
                                  float objIntensity,
                                  float ambientIntensity,
                                  float lightIntensity,
                                  float shininess,
                                  SpotLightSource light,
                                  bool blinn,
                                  vec3 viewDir)
{
    vec3 lightDir = normalize(light.position - f_pos);

    // compute distance and attenuation
    float distVal = length(light.position - f_pos);
    float atten = computeAttenuation(light.attenuation, distVal);

    // spotlight intensity
    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intens = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);

    // costheta for direction
    float costheta = max(dot(normal, lightDir), 0.0);
    // specular
    float specAngle = 0.0;
    if(blinn)
    {
        vec3 halfwayDir = normalize(lightDir + viewDir);
        specAngle = max(dot(normal, halfwayDir), 0.0);
    }
    else
    {
       vec3 reflectDir = reflect(-lightDir, normal);
       specAngle = max(dot(viewDir, reflectDir), 0.0);
    }
    float spec = pow(specAngle, shininess);
    //
    float diffIntensity = lightIntensity * costheta * objIntensity;
    float specIntensity = lightIntensity * spec * objIntensity;
    diffIntensity = diffIntensity * atten * intens;
    specIntensity = specIntensity * atten * intens;
    return (diffIntensity + specIntensity + ambientIntensity);
}
float computeAttenuation(vec3 att, float distVal)
{
    // f_att = min(\frac{1}{c_1 + c_2{\times}d_L + c_3{\times}d^2_{L}} , 1)
    float distSqr = distVal * distVal;
    float att1 = distVal * att.y;
    float att2 = distSqr * att.z;
    float result = att.x + att2 + att1;
    result = 1 / result;
    return min(result, 1.0);
}
"""
quadVshaderDir = """
#version 330 core

// specify input
in vec3 aPos;
in vec3 aNormal;
in vec2 aTexCoord;
in vec3 aTangent;
in vec3 aBiTangent;

// specify output
out vec3 FragPos;
out vec2 TexCoords;
out vec3 TangentViewPos;
out vec3 TangentFragPos;
out DLight {
    vec3 TangentLightDir;
    vec3 color;
} dirLight;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform vec3 viewPos;

struct DirLightSource {

    vec3 direction;
    vec3 color;

};
uniform DirLightSource light;


// function

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    TexCoords = aTexCoord;

    // create tangent bitangent normal matrix for
    // transforming light coordinates
    mat3 normMat = transpose(inverse(mat3(model)));
    vec3 TanV = normalize(normMat * aTangent);
    vec3 NormV = normalize(normMat * aNormal);
    TanV = normalize(TanV - dot(TanV, NormV) * NormV);
    vec3 BiTanV = cross(TanV, NormV);

    // computing light position and view position in tangent space
    mat3 TBN = transpose(mat3(TanV, BiTanV, NormV));
    TangentViewPos = TBN * viewPos;
    TangentFragPos = TBN * FragPos;
    dirLight.TangentLightDir = TBN * light.direction;
    dirLight.color = light.color;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""
quadFshaderDir = """
#version 330 core
in vec3 FragPos;
in vec2 TexCoords;
in vec3 TangentViewPos;
in vec3 TangentFragPos;
in DLight {
    vec3 TangentLightDir;
    vec3 color;
} dlight;
out vec4 FragColor;

struct DirLightSource {

    vec3 direction;
    vec3 color;

};


struct PtmMaterial {

    sampler2D diffuseMap; // object colors per fragment
    sampler2D normalMap1; // normals per vertex
    sampler2D normalMap2; // normals per vertex
    sampler2D normalMap3; // normals per vertex
    float shininess;

};
uniform PtmMaterial material;
uniform vec3 ambient;

float computeColorPerChannelDir(vec3 normal,
                                float objIntensity,
                                float ambientIntensity,
                                float lightIntensity,
                                float shininess,
                                DirLightSource light,
                                vec3 viewDir);

void main(void)
{
    // obtain normal map from texture [0,1]
    vec3 normalr = texture(material.normalMap1, TexCoords).rgb;
    vec3 normalg = texture(material.normalMap2, TexCoords).rgb;
    vec3 normalb = texture(material.normalMap3, TexCoords).rgb;
    // transform vector to range [-1,1]
    normalr = normalize((normalr * 2.0) - 1.0);
    normalg = normalize((normalg * 2.0) - 1.0);
    normalb = normalize((normalb * 2.0) - 1.0);

    // make light
    DirLightSource light;
    light.direction = dlight.TangentLightDir;
    light.color = dlight.color;

    // get diffuse color for object
    vec3 viewDirection = normalize(TangentViewPos - TangentFragPos);
    vec3 objColor = texture(material.diffuseMap, TexCoords).rgb;
    float red = computeColorPerChannelDir(normalr, objColor.x, ambient.x,
                                          light.color.x,
                                       material.shininess, light,
                                       viewDirection);
    float green = computeColorPerChannelDir(normalg, objColor.y, ambient.y,
                                          light.color.y,
                                       material.shininess, light,
                                       viewDirection);
    float blue = computeColorPerChannelDir(normalb, objColor.z, ambient.z,
                                          light.color.z,
                                       material.shininess, light,
                                       viewDirection);
    FragColor = vec4(red, green, blue, 1.0);
}

float computeColorPerChannelDir(vec3 normal,
                             float objIntensity,
                             float ambientIntensity,
                             float lightIntensity,
                             float shininess,
                             DirLightSource light,
                             vec3 viewDir)
{
    vec3 lightDir = normalize(-light.direction);
    // costheta for direction
    float costheta = max(dot(normal, lightDir), 0.0);
    // specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    float diffIntensity = lightIntensity * costheta * objIntensity;
    float specIntensity = lightIntensity * spec * objIntensity;
    return (diffIntensity + ambientIntensity + specIntensity);
}
"""

quadVshaderPoint = """
#version 330 core

// specify input
in vec3 aPos;
in vec3 aNormal;
in vec2 aTexCoord;
in vec3 aTangent;
in vec3 aBiTangent;

// specify output
out vec3 FragPos;
out vec2 TexCoords;
out vec3 TangentViewPos;
out vec3 TangentFragPos;
out PLight {
    vec3 TangentLightPos;
    vec3 color;
    vec3 attenuation;
} pointLight;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform vec3 viewPos;

struct PointLightSource {

    vec3 position;
    vec3 color;
    vec3 attenuation;

};
uniform PointLightSource light;


// function

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    TexCoords = aTexCoord;

    // create tangent bitangent normal matrix for
    // transforming light coordinates
    mat3 normMat = transpose(inverse(mat3(model)));
    vec3 TanV = normalize(normMat * aTangent);
    vec3 NormV = normalize(normMat * aNormal);
    TanV = normalize(TanV - dot(TanV, NormV) * NormV);
    vec3 BiTanV = cross(TanV, NormV);

    // computing light position and view position in tangent space
    mat3 TBN = transpose(mat3(TanV, BiTanV, NormV));
    TangentViewPos = TBN * viewPos;
    TangentFragPos = TBN * FragPos;
    pointLight.TangentLightPos = TBN * light.position;
    pointLight.color = light.color;
    pointLight.attenuation = light.attenuation;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}

"""

quadFshaderPoint = """
#version 330 core
in vec3 FragPos;
in vec2 TexCoords;
in vec3 TangentViewPos;
in vec3 TangentFragPos;
in PLight {
    vec3 TangentLightPos;
    vec3 color;
    vec3 attenuation;
} pointLight;

out vec4 FragColor;

struct PointLight {
    vec3 position;
    vec3 color;
    vec3 attenuation;
};

struct PtmMaterial {

    sampler2D diffuseMap; // object colors per fragment
    sampler2D normalMap1; // normals per vertex
    sampler2D normalMap2; // normals per vertex
    sampler2D normalMap3; // normals per vertex
    float shininess;

};

uniform PtmMaterial material;
uniform vec3 ambient;
uniform vec3 viewPos;

float computeColorPerChannelPoint(vec3 normal,
                                  float objIntensity,
                                  float ambientIntensity,
                                  float lightIntensity,
                                  float shininess,
                                  PointLight light,
                                  vec3 viewDir);
float computeAttenuation(vec3 att, float distVal);

void main(void)
{
    // obtain normal map from texture [0,1]
    vec3 normalr = texture(material.normalMap1, TexCoords).rgb;
    vec3 normalg = texture(material.normalMap2, TexCoords).rgb;
    vec3 normalb = texture(material.normalMap3, TexCoords).rgb;
    // transform vector to range [-1,1]
    normalr = normalize((normalr * 2.0) - 1.0);
    normalg = normalize((normalg * 2.0) - 1.0);
    normalb = normalize((normalb * 2.0) - 1.0);

    // get diffuse color for object
    vec3 viewDirection = normalize(TangentViewPos - TangentFragPos);
    vec3 objColor = texture(material.diffuseMap, TexCoords).rgb;
    PointLight light;
    light.position = pointLight.TangentLightPos;
    light.color = pointLight.color;
    light.attenuation = pointLight.attenuation;

    float red = computeColorPerChannelPoint(normalr, objColor.x, ambient.x,
                                            light.color.x,
                                       material.shininess, light,
                                       viewDirection);
    float green = computeColorPerChannelPoint(normalg, objColor.y, ambient.y,
                                            light.color.y,
                                       material.shininess, light,
                                       viewDirection);
    float blue = computeColorPerChannelPoint(normalb, objColor.z, ambient.z,
                                            light.color.z,
                                       material.shininess, light,
                                       viewDirection);
    FragColor = vec4(red, green, blue, 1.0);
}

float computeColorPerChannelPoint(vec3 normal,
                                  float objIntensity,
                                  float ambientIntensity,
                                  float lightIntensity,
                                  float shininess,
                                  PointLight light,
                                  vec3 viewDir)
{
    vec3 lightDir = normalize(light.position - TangentFragPos);

    // compute distance and attenuation
    float distVal = length(light.position - TangentFragPos);
    float atten = computeAttenuation(light.attenuation, distVal);

    // costheta for direction
    float costheta = max(dot(normal, lightDir), 0.0);
    // specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    //
    float diffIntensity = lightIntensity * costheta * objIntensity;
    float specIntensity = lightIntensity * spec * objIntensity;
    diffIntensity = diffIntensity * atten;
    specIntensity = specIntensity * atten;
    return (diffIntensity + specIntensity + ambientIntensity);

}
float computeAttenuation(vec3 att, float distVal)
{
    // f_att = min(\frac{1}{c_1 + c_2{\times}d_L + c_3{\times}d^2_{L}} , 1)
    float distSqr = distVal * distVal;
    float att1 = distVal * att.y;
    float att2 = distSqr * att.z;
    float result = att.x + att2 + att1;
    result = 1 / result;
    return min(result, 1.0);
}
"""


quadVshaderSpot = """
#version 330 core

// specify input
in vec3 aPos;
in vec3 aNormal;
in vec2 aTexCoord;
in vec3 aTangent;
in vec3 aBiTangent;

// specify output
out vec3 FragPos;
out vec2 TexCoords;
out vec3 TangentViewPos;
out vec3 TangentFragPos;
out SLight {
    vec3 TangentLightDir;
    vec3 TangentLightPos;
    float cutOff;
    float outerCutOff;
    vec3 attenuation;
    vec3 color;
} spotLight;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform vec3 viewPos;

struct SpotLight {

    vec3 position;
    vec3 direction;
    float cutOff;
    float outerCutOff;
    vec3 attenuation;
    vec3 color;

};
uniform SpotLight light;


// function

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    TexCoords = aTexCoord;

    // create tangent bitangent normal matrix for
    // transforming light coordinates
    mat3 normMat = transpose(inverse(mat3(model)));
    vec3 TanV = normalize(normMat * aTangent);
    vec3 NormV = normalize(normMat * aNormal);
    TanV = normalize(TanV - dot(TanV, NormV) * NormV);
    vec3 BiTanV = cross(TanV, NormV);

    // computing light position and view position in tangent space
    mat3 TBN = transpose(mat3(TanV, BiTanV, NormV));
    spotLight.TangentLightPos = TBN * light.position;
    TangentViewPos = TBN * viewPos;
    TangentFragPos = TBN * FragPos;
    spotLight.TangentLightDir = TBN * light.direction;
    spotLight.cutOff = light.cutOff;
    spotLight.outerCutOff = light.outerCutOff;
    spotLight.attenuation = light.attenuation;
    spotLight.color = light.color;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

quadFshaderSpot = """
#version 330 core
in vec3 FragPos;
in vec2 TexCoords;
in vec3 TangentViewPos;
in vec3 TangentFragPos;
in SLight {
    vec3 TangentLightPos;
    vec3 TangentLightDir;
    vec3 color;
    vec3 attenuation;
    float cutOff;
    float outerCutOff;
} spotLight;

struct SpotLightSource {

    vec3 position;
    vec3 direction;
    float cutOff;
    float outerCutOff;
    vec3 attenuation;
    vec3 color;

};

out vec4 FragColor;

struct PtmMaterial {

    sampler2D diffuseMap; // object colors per fragment
    sampler2D normalMap1; // normals per vertex
    sampler2D normalMap2; // normals per vertex
    sampler2D normalMap3; // normals per vertex
    float shininess;

};

uniform PtmMaterial material;
uniform vec3 ambient;
uniform vec3 viewPos;

float computeColorPerChannelSpot(vec3 normal,
                                 float objIntensity,
                                 float ambientIntensity,
                                 float lightIntensity,
                                 float shininess,
                                 SpotLightSource light,
                                 vec3 viewDir);
float computeAttenuation(vec3 att, float distVal);

void main(void)
{
    // obtain normal map from texture [0,1]
    vec3 normalr = texture(material.normalMap1, TexCoords).rgb;
    vec3 normalg = texture(material.normalMap2, TexCoords).rgb;
    vec3 normalb = texture(material.normalMap3, TexCoords).rgb;
    // transform vector to range [-1,1]
    normalr = normalize((normalr * 2.0) - 1.0);
    normalg = normalize((normalg * 2.0) - 1.0);
    normalb = normalize((normalb * 2.0) - 1.0);

    // spot light 
    SpotLightSource light;
    light.position = spotLight.TangentLightPos;
    light.direction = spotLight.TangentLightDir;
    light.color = spotLight.color;
    light.attenuation = spotLight.attenuation;
    light.cutOff = spotLight.cutOff;
    light.outerCutOff = spotLight.outerCutOff;

    // get diffuse color for object
    vec3 viewDirection = normalize(TangentViewPos - TangentFragPos);
    vec3 objColor = texture(material.diffuseMap, TexCoords).rgb;

    float red = computeColorPerChannelSpot(normalr, objColor.x, ambient.x,
                                            light.color.x,
                                       material.shininess, light,
                                       viewDirection);
    float green = computeColorPerChannelSpot(normalg, objColor.y, ambient.y,
                                            light.color.y,
                                       material.shininess, light,
                                       viewDirection);
    float blue = computeColorPerChannelSpot(normalb, objColor.z, ambient.z,
                                            light.color.z,
                                       material.shininess, light,
                                       viewDirection);
    FragColor = vec4(red, green, blue, 1.0);
}

float computeColorPerChannelSpot(vec3 normal,
                                 float objIntensity,
                                 float ambientIntensity,
                                 float lightIntensity,
                                 float shininess,
                                 SpotLightSource light,
                                 vec3 viewDir)
{
    vec3 lightDir = normalize(light.position - TangentFragPos);

    // compute distance and attenuation
    float distVal = length(light.position - TangentFragPos);
    float atten = computeAttenuation(light.attenuation, distVal);

    // spotlight intensity
    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intens = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);

    // costheta for direction
    float costheta = max(dot(normal, lightDir), 0.0);
    // specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    //
    float diffIntensity = lightIntensity * costheta * objIntensity;
    float specIntensity = lightIntensity * spec * objIntensity;
    diffIntensity = diffIntensity * atten * intens;
    specIntensity = specIntensity * atten * intens;
    return (diffIntensity + specIntensity + ambientIntensity);
}
float computeAttenuation(vec3 att, float distVal)
{
    // f_att = min(\frac{1}{c_1 + c_2{\times}d_L + c_3{\times}d^2_{L}} , 1)
    float distSqr = distVal * distVal;
    float att1 = distVal * att.y;
    float att2 = distSqr * att.z;
    float result = att.x + att2 + att1;
    result = 1 / result;
    return min(result, 1.0);
}
"""

quadFshader = """
#version 330 core
in vec3 FragPos;
in vec2 TexCoords;
in vec3 TangentLightPos;
in vec3 TangentViewPos;
in vec3 TangentFragPos;

out vec4 FragColor;

uniform sampler2D diffuseMap; // object colors per fragment
uniform sampler2D normalMap; // normals per vertex

uniform float ambientCoeff;
uniform float shininess;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 viewPos;

void main()
{
    // obtain normal map from texture [0,1]
    vec3 normal = texture(normalMap, TexCoords).rgb;

    // transform vector to range [-1,1]
    normal = normalize((normal * 2.0) - 1.0);

    // get diffuse color
    vec3 color = texture(diffuseMap, TexCoords).rgb;

    // ambient color for object
    vec3 ambientColor = color * ambientCoeff;

    // costheta
    vec3 lightDir = normalize(TangentLightPos - TangentFragPos);
    float costheta = dot(lightDir, normal);
    costheta = max(costheta, 0.0);
    vec3 diffuseColor = color * costheta;

    // specular
    vec3 viewDir = normalize(TangentViewPos - TangentFragPos);
    // vec3 reflectDir = reflect(-lightDir, normal);
    vec3 halfway = normalize(lightDir + viewDir);
    float specAngle = dot(normal, halfway);
    specAngle = max(specAngle, 0.0);
    specAngle = pow(specAngle, shininess);
    vec3 specularColor = lightColor * specAngle;

    // final fragment color
    FragColor = vec4(ambientColor + diffuseColor + specularColor, 1.0);
}
"""

quadFshaderPerChannelSpot = """
#version 330 core

in vec3 FragPos;
in vec2 TexCoords;
in vec3 TangentLightPos;
in vec3 TangentLightDir;
in vec3 TangentViewPos;
in vec3 TangentFragPos;

out vec4 FragColor;

struct PtmMaterial {

    uniform sampler2D diffuseMap; // object colors per fragment
    uniform sampler2D normalMap1; // normals per vertex
    uniform sampler2D normalMap2; // normals per vertex
    uniform sampler2D normalMap3; // normals per vertex
    uniform float shininess;

};


uniform PtmMaterial material;

float computeDiffColorPerChannel(vec3 normal, vec3 lightDir, float intensity);
float computeSpecColorPerChannel(vec3 normal, vec3 dirVec,
                                 float shininess, float intensity);
vec3 computeSpecColor(vec3 normal1, vec3 normal2, vec3 normal3, vec3 dirVec,
                      float shininess, vec3 color);
vec3 computeDiffColor(vec3 normal1, vec3 normal2, vec3 normal3, vec3 lightDir,
                      vec3 color);
float computeAttenuation(vec3 att, float distVal);



void main()
{
    // obtain normal map from texture [0,1]
    vec3 normalr = texture(material.normalMap1, TexCoords).rgb;
    vec3 normalg = texture(material.normalMap2, TexCoords).rgb;
    vec3 normalb = texture(material.normalMap3, TexCoords).rgb;

    // get diffuse color for object
    vec3 color = texture(material.diffuseMap, TexCoords).rgb;

    // attenuation
    float distanceLightFrag = length(TangentLightPos - TangentFragPos);
    float att = computeAttenuation(light.attenuation, distanceLightFrag);

    // ambient color for object
    vec3 ambientColor = color * ambientCoeff;

    // simple light direction
    vec3 tanLightDir = normalize(TangentLightPos - TangentFragPos);

    // spotlight cone theta
    float theta = dot(tanLightDir, TangentLightDir);

    // costheta
    vec3 diffuseColor = computeDiffColor(normalr, normalg, normalb,
                                         tanLightDir, color) * att;
    // specular
    vec3 viewDir = normalize(TangentViewPos - TangentFragPos);
    // vec3 reflectDir = reflect(-tanLightDir, normal);
    vec3 halfway = normalize(tanLightDir + viewDir);
    vec3 specularColor = computeSpecColor(normalr, normalg, normalb, halfway,
                                          shininess, lightColor) * att;
    // final fragment color
    FragColor = vec4(ambientColor + diffuseColor + specularColor, 1.0);
}


float computeAttenuation(vec3 att, float distVal)
{
    // f_att = min(\frac{1}{c_1 + c_2{\times}d_L + c_3{\times}d^2_{L}} , 1)
    float distSqr = distVal * distVal;
    float att1 = distVal * att.y;
    float att2 = distSqr * att.z;
    float result = att.x + att2 + att1;
    result = 1 / result;
    return min(result, 1.0);
}

float computeDiffColorPerChannel(vec3 normal, vec3 lightDir, float intensity)
{
    // transform vector to range [-1,1]
    normal = normalize((normal * 2.0) - 1.0);
    float costheta = dot(lightDir, normal);
    costheta = max(costheta, 0.0);
    return costheta * intensity;
}
vec3 computeDiffColor(vec3 normal1, vec3 normal2, vec3 normal3, vec3 lightDir,
                      vec3 color)
{
    vec3 diffuseColor = vec3(1.0);
    diffuseColor.x = computeDiffColorPerChannel(normal1, lightDir, color.x);
    diffuseColor.y = computeDiffColorPerChannel(normal2, lightDir, color.y);
    diffuseColor.z = computeDiffColorPerChannel(normal3, lightDir, color.z);
    return diffuseColor;
}

float computeSpecColorPerChannel(vec3 normal, vec3 dirVec,
                                 float shininess, float intensity)
{
    normal = normalize((normal * 2.0) - 1.0);
    float specAngle = dot(normal, dirVec);
    specAngle = max(specAngle, 0.0);
    specAngle = pow(specAngle, shininess);
    return specAngle * intensity;
}
vec3 computeSpecColor(vec3 normal1, vec3 normal2, vec3 normal3, vec3 dirVec,
                      float shininess, vec3 color)
{
    vec3 specularColor = vec3(1.0);
    specularColor.x = computeSpecColorPerChannel(normal1, dirVec,
                                                 shininess, lightColor.x);
    specularColor.y = computeSpecColorPerChannel(normal2, dirVec,
                                                 shininess, lightColor.y);
    specularColor.z = computeSpecColorPerChannel(normal3, dirVec,
                                                 shininess, lightColor.z);
    return specularColor;
}
"""

quadFshaderPerChannelMulti = """
#version 330 core

in vec3 FragPos;
in vec2 TexCoords;
in vec3 TangentLightPos;
in vec3 TangentLightDir;
in vec3 TangentViewPos;
in vec3 TangentFragPos;

out vec4 FragColor;

struct PtmMaterial {

uniform sampler2D diffuseMap; // object colors per fragment
uniform sampler2D normalMap1; // normals per vertex
uniform sampler2D normalMap2; // normals per vertex
uniform sampler2D normalMap3; // normals per vertex
uniform float shininess;

};

struct DirLight {
    vec3 directional;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct PointLight {

    vec3 position;
    vec3 attenuation;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

};

struct SpotLight {

    vec3 position;
    vec3 direction;
    float cutOff;
    float outerCutOff;
    vec3 attenuation;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

};

uniform float ambientCoeff;
uniform vec3 attc;
uniform float cutOff;

uniform vec3 lightPos;
uniform vec3 lightDirection;
uniform vec3 lightColor;
uniform vec3 viewPos;

float computeDiffColorPerChannel(vec3 normal, vec3 lightDir, float intensity);
float computeSpecColorPerChannel(vec3 normal, vec3 dirVec,
                                 float shininess, float intensity);
vec3 computeSpecColor(vec3 normal1, vec3 normal2, vec3 normal3, vec3 dirVec,
                      float shininess, vec3 color);
vec3 computeDiffColor(vec3 normal1, vec3 normal2, vec3 normal3, vec3 lightDir,
                      vec3 color);
float computeAttenuation(vec3 att, float distVal);



void main()
{
    // obtain normal map from texture [0,1]
    vec3 normalr = texture(normalMap1, TexCoords).rgb;
    vec3 normalg = texture(normalMap2, TexCoords).rgb;
    vec3 normalb = texture(normalMap3, TexCoords).rgb;

    // get diffuse color for object
    vec3 color = texture(diffuseMap, TexCoords).rgb;

    // attenuation
    float distanceLightFrag = length(TangentLightPos - TangentFragPos);
    float att = computeAttenuation(attc, distanceLightFrag);

    // ambient color for object
    vec3 ambientColor = color * ambientCoeff;

    // simple light direction
    vec3 tanLightDir = normalize(TangentLightPos - TangentFragPos);

    // spotlight cone theta
    float theta = dot(tanLightDir, TangentLightDir);

    // costheta
    vec3 diffuseColor = computeDiffColor(normalr, normalg, normalb,
                                         tanLightDir, color) * att;
    // specular
    vec3 viewDir = normalize(TangentViewPos - TangentFragPos);
    // vec3 reflectDir = reflect(-tanLightDir, normal);
    vec3 halfway = normalize(tanLightDir + viewDir);
    vec3 specularColor = computeSpecColor(normalr, normalg, normalb, halfway,
                                          shininess, lightColor) * att;
    // final fragment color
    FragColor = vec4(ambientColor + diffuseColor + specularColor, 1.0);
}


float computeAttenuation(vec3 att, float distVal)
{
    // f_att = min(\frac{1}{c_1 + c_2{\times}d_L + c_3{\times}d^2_{L}} , 1)
    float distSqr = distVal * distVal;
    float att1 = distVal * att.y;
    float att2 = distSqr * att.z;
    float result = att.x + att2 + att1;
    result = 1 / result;
    return min(result, 1);
}

float computeDiffColorPerChannel(vec3 normal, vec3 lightDir, float intensity)
{
    // transform vector to range [-1,1]
    normal = normalize((normal * 2.0) - 1.0);
    float costheta = dot(lightDir, normal);
    costheta = max(costheta, 0.0);
    return costheta * intensity;
}
vec3 computeDiffColor(vec3 normal1, vec3 normal2, vec3 normal3, vec3 lightDir,
                      vec3 color)
{
    vec3 diffuseColor = vec3(1.0);
    diffuseColor.x = computeDiffColorPerChannel(normal1, lightDir, color.x);
    diffuseColor.y = computeDiffColorPerChannel(normal2, lightDir, color.y);
    diffuseColor.z = computeDiffColorPerChannel(normal3, lightDir, color.z);
    return diffuseColor;
}

float computeSpecColorPerChannel(vec3 normal, vec3 dirVec,
                                 float shininess, float intensity)
{
    normal = normalize((normal * 2.0) - 1.0);
    float specAngle = dot(normal, dirVec);
    specAngle = max(specAngle, 0.0);
    specAngle = pow(specAngle, shininess);
    return specAngle * intensity;
}
vec3 computeSpecColor(vec3 normal1, vec3 normal2, vec3 normal3, vec3 dirVec,
                      float shininess, vec3 color)
{
    vec3 specularColor = vec3(1.0);
    specularColor.x = computeSpecColorPerChannel(normal1, dirVec,
                                                 shininess, lightColor.x);
    specularColor.y = computeSpecColorPerChannel(normal2, dirVec,
                                                 shininess, lightColor.y);
    specularColor.z = computeSpecColorPerChannel(normal3, dirVec,
                                                 shininess, lightColor.z);
    return specularColor;
}
"""


lampVshader = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

lampFshader = """
#version 330 core
out vec4 FragColor;

uniform lowp vec3 lightColor;

void main()
{
    FragColor = vec4(lightColor, 0.1);
}
"""

rgbptmVshader = """
#version 330 core
layout (location = 0) in vec3 aPos;

layout (location = 1) in vec3 acoeff13r;
layout (location = 2) in vec3 acoeff46r;

layout (location = 3) in vec3 acoeff13g;
layout (location = 4) in vec3 acoeff46g;

layout (location = 5) in vec3 acoeff13b;
layout (location = 6) in vec3 acoeff46b;

// six coeff per channel

out vec3 DiffColor;
out vec3 NormalR;
out vec3 NormalG;
out vec3 NormalB;
out vec3 FragPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

float computeLuPerChannel(float c0, float c1, float c2, float c3, float c4);
float computeLvPerChannel(float c0, float c1, float c2, float c3, float c4);
vec3 computeNormalPerChannel(float c0, float c1, float c2, float c3, float c4);
float computeDiffuseColorPerChannel(float c0, float c1, float c2, float c3,
                                    float c4, float c5, float Lu, float Lv);

vec3 computeDiffuseColor(float c0r, float c1r,
                         float c2r, float c3r,
                         float c4r, float c5r,
                         float c0g, float c1g,
                         float c2g, float c3g,
                         float c4g, float c5g,
                         float c0b, float c1b,
                         float c2b, float c3b,
                         float c4b, float c5b);

void main()
{
    vec3 surfaceNormalR = computeNormalPerChannel(
        coeff13r.x, coeff13r.y, coeff13r.z, coeff46r.x, coeff46r.y
    );
    vec3 sugfaceNormalG = computeNormalPerChannel(
        coeff13g.x, coeff13g.y, coeff13g.z, coeff46g.x, coeff46g.y
    );
    vec3 surfaceNormalB = computeNormalPerChannel(
        coeff13b.x, coeff13b.y, coeff13b.z, coeff46b.x, coeff46b.y
    );
    vec3 diffColor = computeDiffuseColor(
        coeff13r.x, coeff13r.y, coeff13r.z, coeff46r.x, coeff46r.y, coeff46r.z,
        coeff13g.x, coeff13g.y, coeff13g.z, coeff46g.x, coeff46g.y, coeff46g.z,
        coeff13b.x, coeff13b.y, coeff13b.z, coeff46b.x, coeff46b.y, coeff46b.z
        );
    DiffColor = diffColor;
    NormalR = surfaceNormalR;
    NormalG = surfaceNormalG;
    NormalB = surfaceNormalB;
    
    FragPos = vec3(model * vec4(aPos, 1.0));
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}


float computeLuPerChannel(float c0, float c1, float c2, float c3, float c4)
{
    // taken directly from the paper of Tom Malzbender, Dan Gelb, Hans Wolters
    // http://www.hpl.hp.com/ptm
    return ((c2 * c4) - (2 * c1 * c3)) / ((4 * c0 * c1) - (c2 * c2));
}

float computeLvPerChannel(float c0, float c1, float c2, float c3, float c4)
{ 
    // taken directly from the paper of Tom Malzbender, Dan Gelb, Hans Wolters
    // http://www.hpl.hp.com/ptm
    return ((c2 * c3) - (2 * c0 * c4)) / ((4 * c0 * c1) - (c2 * c2));
}

vec3 computeNormalPerChannel(float c0, float c1, float c2, float c3, float c4)
{
    // taken directly from the paper of Tom Malzbender, Dan Gelb, Hans Wolters
    // http://www.hpl.hp.com/ptm
    float Lu = computeLuPerChannel(c0,c1,
                                   c2, c3,
                                   c4);
    float Lv = computeLvPerChannel(c0, c1,
                                   c2, c3,
                                   c4);

    return vec3(Lu, Lv, sqrt(1 - (Lu * Lu) - (Lv * Lv)));
}

float computeDiffuseColorPerChannel(float c0, float c1, float c2, float c3,
                         float c4, float c5, float Lu, float Lv)
{
    // taken directly from the paper of Tom Malzbender, Dan Gelb, Hans Wolters
    // http://www.hpl.hp.com/ptm
    float term1 = c0 * Lu * Lu;
    float term2 = c1 * Lv * Lv;
    float term3 = c2 * Lu * Lv;
    float term4 = c3 * Lu;
    float term5 = c4 * Lv;
    float term6 = c5;
    return term1 + term2 + term3 + term4 + term5 + term6;
}
vec3 computeDiffuseColor(float c0r, float c1r,
                         float c2r, float c3r,
                         float c4r, float c5r,
                         float c0g, float c1g,
                         float c2g, float c3g,
                         float c4g, float c5g,
                         float c0b, float c1b,
                         float c2b, float c3b,
                         float c4b, float c5b)
{
    float LuR = computeLuPerChannel(
        c0r, c1r, c2r, c3r, c4r
    );
    float LvR = computeLvPerChannel(
        c0r, c1r, c2r, c3r, c4r
    );
    float LuG = computeLuPerChannel(
        c0g, c1g, c2g, c3g, c4g
    );
    float LvG = computeLvPerChannel(
        c0g, c1g, c2g, c3g, c4g
    );
    float LuB = computeLuPerChannel(
        c0b, c1b, c2b, c3b, c4b
    );
    float LvB = computeLvPerChannel(
        c0b, c1b, c2b, c3b, c4b
    );
    return vec3(
    computeDiffuseColorPerChannel(c0r, c1r, c2r, c3r, c4r, c5r, LuR, LvR),
    computeDiffuseColorPerChannel(c0g, c1g, c2g, c3g, c4g, c5g, LuG, LvG),
    computeDiffuseColorPerChannel(c0b, c1b, c2b, c3b, c4b, c5b, LuB, LvB)
    );
}
"""

rgbptmFshader = """
#version 330 core

out vec4 FragColor;

in vec3 FragPos;

// six coeff per channel

in vec3 NormalR;
in vec3 NormalG;
in vec3 NormalB;

in vec3 DiffColor;

uniform vec3 lightPos;
uniform vec3 viewPos;

uniform vec3 lightColor;
uniform float shininess;

uniform vec3 attc;

uniform vec3 diffuseCoeffs;
uniform vec3 specularCoeffs;
uniform vec3 ambientCoeffs;

uniform bool blinn;

float computeAttenuation(vec3 atten, float distVal);

float computeFragColorPerChannel(vec3 surfaceNormal, float attc1,
                                float attc2, float attc3,
                                float objDiffuseChannelIntensity,
                                float objDiffuseChannelCoeff,
                                float ambientChannelCoeff,
                                float ambientChannelIntensity,
                                float specularChannelIntensity,
                                float specularChannelCoeff,
                                float lightChannelIntensity,
                                float shininess, vec3 viewerPosition,
                                vec3 fragPos, vec3 lightPosition,
                                bool blinn);

void main()
{

    float redC = computeFragColorPerChannel(
                    NormalR,
                    attc, DiffColor.x,
                    diffuseCoeffs.x,
                    ambientCoeffs.x,
                    lightColor.x,  // ambient channel intensity
                    DiffColor.x,  // specular channel intensity
                    specularCoeffs.x,
                    lightColor.x,
                    shininess,
                    viewPos,
                    FragPos,
                    lightPos,
                    blinn);
    float greenC = computeFragColorPerChannel(
                    NormalG,
                    attc,
                    DiffColor.y,
                    diffuseCoeffs.y,
                    ambientCoeffs.y,
                    lightColor.y,  // ambient channel intensity
                    DiffColor.y,  // specular channel intensity
                    specularCoeffs.y,
                    lightColor.y,  // light channel intensity
                    shininess,
                    viewPos,
                    FragPos,
                    lightPos,
                    blinn);
    float blueC = computeFragColorPerChannel(
                    NormalB,
                    attc,
                    DiffColor.z,
                    diffuseCoeffs.z,
                    ambientCoeffs.z,
                    lightColor.z,  // ambient channel intensity
                    DiffColor.z,  // specular channel intensity
                    specularCoeffs.z,
                    lightColor.z,  // light channel intensity
                    shininess,
                    viewPos,
                    FragPos,
                    lightPos,
                    blinn);
    FragColor = vec4(redC, greenC, blueC, 1.0);
}

float computeAttenuation(vec3 att, float distVal)
{
    // f_att = min(\frac{1}{c_1 + c_2{\times}d_L + c_3{\times}d^2_{L}} , 1)
    float distSqr = distVal * distVal;
    float att1 = distVal * att.y;
    float att2 = distSqr * att.z;
    float result = att.x + att2 + att1;
    result = 1 / result;
    return min(result, 1);
}

float computeFragColorPerChannel(vec3 surfaceNormal, vec3 attc,
                                float objDiffuseChannelIntensity,
                                float objDiffuseChannelCoeff,
                                float ambientChannelCoeff,
                                float ambientChannelIntensity,
                                float specularChannelIntensity,
                                float specularChannelCoeff,
                                float lightChannelIntensity,
                                float shininess, vec3 viewerPosition,
                                vec3 fragPos, vec3 lightPosition,
                                bool blinn)
{
    // blinn phong light illumination
    float ambientTerm = ambientChannelIntensity * ambientChannelCoeff;
    ambientTerm = ambientTerm * objDiffuseChannelIntensity;

    // attenuation
    float distVal = length(lightPosition, fragPos);
    float fattr = computeAttenuation(attc, distVal);

    // lambertian term
    vec3 snormal = normalize(surfaceNormal);
    vec3 lightDirection = normalize(lightPosition - fragPos);
    float costheta = dot(lightDirection, snormal);
    float lambertian = costheta * objDiffuseChannelCoeff;
    lambertian = lambertian * objDiffuseChannelIntensity;
    lambertian = lambertian * lightChannelIntensity;
    lambertian = lambertian * fattr;

    // specular term
    vec3 viewerDirection = normalize(viewerPosition - fragPos);
    float specAngle = 0.0;
    if (blinn)
    {
        vec3 halfway = normalize(lightDirection + viewerDirection);
        specAngle = max(dot(snormal, halfway), 0);
    }else{
        vec3 reflectionDir = reflect(-lightDirection, snormal);
        specAngle = max(dot(viewerDirection, reflectionDir), 0);
    }
    float specTerm = specularChannelIntensity * specularChannelCoeff;
    specTerm = pow(specAngle, shininess) * specTerm;
    return ambientTerm + lambertian + specTerm;
}

"""

shaders = {
    "lambert": {
        "fragment": lambertFshader,
        "vertex": lambertVshader,
        "attribute_info": {
            "stride": None,
            "aPos": {"layout": 0, "size": 3, "offset": 0},
            "aNormal": {"layout": 1, "size": 3, "offset": 3},
            "aTexCoord": {"layout": 2, "size": 2, "offset": 6},
        },
    },
    "phong": {
        "fragment": phongFshader,
        "vertex": phongVshader,
        "attribute_info": {
            "stride": None,
            "aPos": {"layout": 0, "size": 3, "offset": 0},
            "aNormal": {"layout": 1, "size": 3, "offset": 3},
            "aTexCoord": {"layout": 2, "size": 2, "offset": 6},
        },
    },
    "quadDir": {
        "fragment": quadFshaderDir,
        "vertex": quadVshaderDir,
        "attribute_info": {
            "stride": None,
            "aPos": {"layout": 0, "size": 3, "offset": 0},
            "aNormal": {"layout": 1, "size": 3, "offset": 3},
            "aTexCoord": {"layout": 2, "size": 2, "offset": 6},
            "aTangent": {"layout": 3, "size": 3, "offset": 8},
            "aBiTangent": {"layout": 4, "size": 3, "offset": 11},
        },
    },
    "quadPoint": {
        "fragment": quadFshaderPoint,
        "vertex": quadVshaderPoint,
        "attribute_info": {
            "stride": None,
            "aPos": {"layout": 0, "size": 3, "offset": 0},
            "aNormal": {"layout": 1, "size": 3, "offset": 3},
            "aTexCoord": {"layout": 2, "size": 2, "offset": 6},
            "aTangent": {"layout": 3, "size": 3, "offset": 8},
            "aBiTangent": {"layout": 4, "size": 3, "offset": 11},
        },
    },
    "quadSpot": {
        "fragment": quadFshaderSpot,
        "vertex": quadVshaderSpot,
        "attribute_info": {
            "stride": None,
            "aPos": {"layout": 0, "size": 3, "offset": 0},
            "aNormal": {"layout": 1, "size": 3, "offset": 3},
            "aTexCoord": {"layout": 2, "size": 2, "offset": 6},
            "aTangent": {"layout": 3, "size": 3, "offset": 8},
            "aBiTangent": {"layout": 4, "size": 3, "offset": 11},
        },
    },
    "lamp": {
        "fragment": lampFshader,
        "vertex": lampVshader,
        "attribute_info": {
            "stride": None,
            "aPos": {"layout": 0, "size": 3, "offset": 0},
        },
    },
    "rgbcoeff": {
        "fragment": rgbptmFshader,
        "vertex": rgbptmVshader,
        "attribute_info": {
            "stride": None,
            "aPos": {"layout": 0, "size": 3, "offset": 0},
            "acoeff1r": {"layout": 1, "size": 1, "offset": 3},
            "acoeff2r": {"layout": 2, "size": 1, "offset": 4},
            "acoeff3r": {"layout": 3, "size": 1, "offset": 5},
            "acoeff4r": {"layout": 4, "size": 1, "offset": 6},
            "acoeff5r": {"layout": 5, "size": 1, "offset": 7},
            "acoeff6r": {"layout": 6, "size": 1, "offset": 8},
            "acoeff1g": {"layout": 7, "size": 1, "offset": 9},
            "acoeff2g": {"layout": 8, "size": 1, "offset": 10},
            "acoeff3g": {"layout": 9, "size": 1, "offset": 11},
            "acoeff4g": {"layout": 10, "size": 1, "offset": 12},
            "acoeff5g": {"layout": 11, "size": 1, "offset": 13},
            "acoeff6g": {"layout": 12, "size": 1, "offset": 14},
            "acoeff1b": {"layout": 13, "size": 1, "offset": 15},
            "acoeff2b": {"layout": 14, "size": 1, "offset": 16},
            "acoeff3b": {"layout": 15, "size": 1, "offset": 17},
            "acoeff4b": {"layout": 16, "size": 1, "offset": 18},
            "acoeff5b": {"layout": 17, "size": 1, "offset": 19},
            "acoeff6b": {"layout": 18, "size": 1, "offset": 20},
        },
    },
}

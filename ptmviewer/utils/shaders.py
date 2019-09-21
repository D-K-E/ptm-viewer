# author: Kaan Eraslan
# license: see, LICENSE.
# Simple file for holding shaders

objectVshader = """
#version 330 core

// specify input
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aBiTangent;

// specify output
out vec3 FragPos;
out vec2 TexCoords;
out vec3 TangentLightPos;
out vec3 TangentViewPos;
out vec3 TangentFragPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform vec3 lightPos;
uniform vec3 viewPos;

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
    TangentLightPos = TBN * lightPos;
    TangentViewPos = TBN * viewPos;
    TangentFragPos = TBN * FragPos;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

objectFshader = """
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

objectFshaderPerChannel = """
#version 330 core
in vec3 FragPos;
in vec2 TexCoords;
in vec3 TangentLightPos;
in vec3 TangentViewPos;
in vec3 TangentFragPos;

out vec4 FragColor;

uniform sampler2D diffuseMap; // object colors per fragment
uniform sampler2D normalMapR; // normals per vertex
uniform sampler2D normalMapG; // normals per vertex
uniform sampler2D normalMapB; // normals per vertex

uniform float ambientCoeff;
uniform float shininess;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 viewPos;

void main()
{
    // obtain normal map from texture [0,1]
    vec3 normalr = texture(normalMapR, TexCoords).rgb;
    vec3 normalg = texture(normalMapG, TexCoords).rgb;
    vec3 normalb = texture(normalMapB, TexCoords).rgb;

    // transform vector to range [-1,1]
    normalr = normalize((normalr * 2.0) - 1.0);
    normalg = normalize((normalg * 2.0) - 1.0);
    normalb = normalize((normalb * 2.0) - 1.0);

    // get diffuse color
    vec3 color = texture(diffuseMap, TexCoords).rgb;

    // ambient color for object
    vec3 ambientColor = color * ambientCoeff;

    // costheta
    vec3 lightDir = normalize(TangentLightPos - TangentFragPos);
    float costhetaR = dot(lightDir, normalr);
    float costhetaG = dot(lightDir, normalg);
    float costhetaB = dot(lightDir, normalb);
    costhetaR = max(costhetaR, 0.0);
    costhetaG = max(costhetaG, 0.0);
    costhetaB = max(costhetaB, 0.0);
    vec3 diffuseColor = vec3(1.0);
    diffuseColor.x = color.x * costhetaR;
    diffuseColor.y = color.y * costhetaG;
    diffuseColor.z = color.z * costhetaB;

    // specular
    vec3 viewDir = normalize(TangentViewPos - TangentFragPos);
    // vec3 reflectDir = reflect(-lightDir, normal);
    vec3 halfway = normalize(lightDir + viewDir);
    float specAngleR = dot(normalr, halfway);
    float specAngleG = dot(normalg, halfway);
    float specAngleB = dot(normalb, halfway);
    specAngleR = max(specAngleR, 0.0);
    specAngleG = max(specAngleG, 0.0);
    specAngleB = max(specAngleB, 0.0);
    specAngleR = pow(specAngleR, shininess);
    specAngleG = pow(specAngleG, shininess);
    specAngleB = pow(specAngleB, shininess);
    vec3 specularColor = vec3(1.0);
    specularColor.x = lightColor.x * specAngleR;
    specularColor.y = lightColor.y * specAngleG;
    specularColor.z = lightColor.z * specAngleB;

    // final fragment color
    FragColor = vec4(ambientColor + diffuseColor + specularColor, 1.0);
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
    FragColor = vec4(lightColor, 1.0);
}
"""

shaders = {
    "quad": {"fragment": objectFshader, "vertex": objectVshader},
    "quadPerChannel": {
        "fragment": objectFshaderPerChannel,
        "vertex": objectVshader,
    },
    "lamp": {"fragment": lampFshader, "vertex": lampVshader},
}

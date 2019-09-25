attribute vec3 vertexPosition;
attribute vec3 surfaceNormal;
attribute vec2 aTextureCoord;

varying highp vec3 FragPos;
varying highp vec3 Normal;
varying highp vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}

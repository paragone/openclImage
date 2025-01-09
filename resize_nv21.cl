__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void resize_nv21(
    __read_only image2d_t y_input_image,  // 输入 Y 分量
    __read_only image2d_t uv_input_image, // 输入 UV 分量
    __write_only image2d_t y_output_image, // 输出 Y 分量
    __write_only image2d_t uv_output_image, // 输出 UV 分量
    int input_width,                      // 输入图像宽度
    int input_height,                     // 输入图像高度
    int output_width,                     // 输出图像宽度
    int output_height                     // 输出图像高度
) {
    int x = get_global_id(0); // 输出图像的 x 坐标
    int y = get_global_id(1); // 输出图像的 y 坐标

    // 计算输入图像的对应坐标
    float src_x = x * ((float)input_width / output_width);
    float src_y = y * ((float)input_height / output_height);

    // 处理 Y 分量
    if (x < output_width && y < output_height) {
        float y_value = read_imagef(y_input_image, sampler, (float2)(src_x, src_y)).x * 255.0f;
        write_imagef(y_output_image, (int2)(x, y), (float4)(y_value / 255.0f, 0.0f, 0.0f, 0.0f));
    }

    // 处理 UV 分量
    if (x < output_width / 2 && y < output_height / 2) {
        // 读取 UV 分量
        float2 uv_value = read_imagef(uv_input_image, sampler, (float2)(src_x / 2, src_y / 2)).xy * 255.0f;

        // 写入 UV 分量
        write_imagef(uv_output_image, (int2)(x, y), (float4)(uv_value.x / 255.0f, uv_value.y / 255.0f, 0.0f, 0.0f));
    }
}
__kernel void compose_nv21_buffer(
    __global const uchar* input_y1,  // 第一张图像的 Y 通道
    __global const uchar* input_uv1, // 第一张图像的 UV 通道
    __global const uchar* input_y2,  // 第二张图像的 Y 通道
    __global const uchar* input_uv2, // 第二张图像的 UV 通道
    __global uchar* output_y,        // 输出 Y 通道
    __global uchar* output_uv,       // 输出 UV 通道
    int width1,                      // 第一张图像的宽度
    int width2,                      // 第二张图像的宽度
    int height                       // 图像高度
)
{
    int x = get_global_id(0); // 输出图像的 x 坐标
    int y = get_global_id(1); // 输出图像的 y 坐标

    int output_width = width1 + width2; // 输出图像的宽度

    // 确保 y 坐标在图像高度范围内
    if (y >= height) return;

    // 处理 Y 通道
    if (x < width1) {
        // 从第一张图像读取 Y 通道
        output_y[y * output_width + x] = input_y1[y * width1 + x];
    } else {
        // 从第二张图像读取 Y 通道
        output_y[y * output_width + x] = input_y2[y * width2 + (x - width1)];
    }

    // 处理 UV 通道（UV 通道的宽度是 Y 通道的一半）
    if (x % 2 == 0 && y % 2 == 0) {
        int uv_x = x / 2;
        int uv_y = y / 2;

        int output_uv_width = output_width / 2;

        if (uv_x < width1 / 2) {
            // 从第一张图像读取 UV 通道
            output_uv[uv_y * output_uv_width * 2 + uv_x * 2] = input_uv1[uv_y * width1 + uv_x * 2];     // U
            output_uv[uv_y * output_uv_width * 2 + uv_x * 2 + 1] = input_uv1[uv_y * width1 + uv_x * 2 + 1]; // V
        } else {
            // 从第二张图像读取 UV 通道
            output_uv[uv_y * output_uv_width * 2 + uv_x * 2] = input_uv2[uv_y * width2 + (uv_x - width1 / 2) * 2];     // U
            output_uv[uv_y * output_uv_width * 2 + uv_x * 2 + 1] = input_uv2[uv_y * width2 + (uv_x - width1 / 2) * 2 + 1]; // V
        }
    }
}

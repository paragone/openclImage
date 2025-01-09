__kernel void rearrange_nv21(
    __global uchar* input,  // 输入图像数据
    __global uchar* output, // 输出图像数据
    int input_width,        // 输入图像宽度 (7680)
    int input_height,       // 输入图像高度 (1300)
    int output_width,       // 输出图像宽度 (8000)
    int output_height       // 输出图像高度 (1500)
) {
    int x = get_global_id(0); // 当前像素的 x 坐标
    int y = get_global_id(1); // 当前像素的 y 坐标

    // 计算黑边大小
    int side_margin = (output_width - input_width) / 5;       // 左右及中间黑边宽度 (64)
    int top_bottom_margin = (output_height - input_height) / 2; // 上下黑边高度 (100)

    // 计算输出图像的 Y 分量位置
    int out_y_index = y * output_width + x;

    // 计算输出图像的 UV 分量位置
    int out_uv_index = output_width * output_height + (y / 2) * output_width + (x / 2) * 2;

    // 判断是否需要填充黑边
    if (x < side_margin || x >= output_width - side_margin || // 左右黑边
        y < top_bottom_margin || y >= output_height - top_bottom_margin || // 上下黑边
        (x >= side_margin + 1920 && x < side_margin + 1920 + side_margin) || // 第1和第2块之间黑边
        (x >= side_margin + 1920 * 2 + side_margin && x < side_margin + 1920 * 2 + side_margin * 2) || // 第2和第3块之间黑边
        (x >= side_margin + 1920 * 3 + side_margin * 2 && x < side_margin + 1920 * 3 + side_margin * 3)) { // 第3和第4块之间黑边
        // 填充黑边
        output[out_y_index] = 0; // Y 分量黑边
        if (x % 2 == 0 && y % 2 == 0) {
            output[out_uv_index] = 128;     // U 分量黑边
            output[out_uv_index + 1] = 128; // V 分量黑边
        }
    } else {
        // 计算输入图像的块索引
        int block_width = 1920; // 每个块的宽度
        int in_x, in_y;         // 输入图像的像素坐标

        if (x < side_margin + block_width) {
            // 第 3 块（原图的第 3 块，索引 2）
            in_x = x - side_margin + block_width * 2;
        } else if (x < side_margin + block_width * 2 + side_margin) {
            // 第 4 块（原图的第 4 块，索引 3）
            in_x = x - side_margin - block_width - side_margin + block_width * 3;
        } else if (x < side_margin + block_width * 3 + side_margin * 2) {
            // 第 1 块（原图的第 1 块，索引 0）
            in_x = x - side_margin - block_width * 2 - side_margin * 2 + block_width * 0;
        } else {
            // 第 2 块（原图的第 2 块，索引 1）
            in_x = x - side_margin - block_width * 3 - side_margin * 3 + block_width * 1;
        }

        in_y = y - top_bottom_margin; // 输入图像的 y 坐标

        // 计算输入图像的 Y 分量索引
        int in_y_index = in_y * input_width + in_x;

        // 计算输入图像的 UV 分量索引
        int in_uv_index = input_width * input_height + (in_y / 2) * input_width + (in_x / 2) * 2;

        // 复制 Y 分量
        output[out_y_index] = input[in_y_index];

        // 复制 UV 分量
        if (x % 2 == 0 && y % 2 == 0) {
            output[out_uv_index] = input[in_uv_index];         // U 分量
            output[out_uv_index + 1] = input[in_uv_index + 1]; // V 分量
        }
    }
}
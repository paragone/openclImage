__kernel void resize_rgb_bilinear(
    __global const uchar* input, 
    __global uchar* output, 
    int input_width, 
    int input_height, 
    int output_width, 
    int output_height) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float x_ratio = (float)(input_width - 1) / output_width;
    float y_ratio = (float)(input_height - 1) / output_height;

    float x_diff = x_ratio * x - (int)(x_ratio * x);
    float y_diff = y_ratio * y - (int)(y_ratio * y);

    int x_l = (int)(x_ratio * x);
    int x_h = x_l + 1;
    int y_l = (int)(y_ratio * y);
    int y_h = y_l + 1;

    int index_l_l = (y_l * input_width + x_l) * 3;
    int index_l_h = (y_l * input_width + x_h) * 3;
    int index_h_l = (y_h * input_width + x_l) * 3;
    int index_h_h = (y_h * input_width + x_h) * 3;

    for (int i = 0; i < 3; i++) {
        float value = (input[index_l_l + i] * (1 - x_diff) * (1 - y_diff) +
                       input[index_l_h + i] * x_diff * (1 - y_diff) +
                       input[index_h_l + i] * (1 - x_diff) * y_diff +
                       input[index_h_h + i] * x_diff * y_diff);

        output[(y * output_width + x) * 3 + i] = (uchar)value;
    }
}
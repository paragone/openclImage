#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_ERROR(err, msg)                               \
    if (err != CL_SUCCESS)                                  \
    {                                                       \
        fprintf(stderr, "%s (Error code: %d)\n", msg, err); \
        exit(EXIT_FAILURE);                                 \
    }

// 读取文件到内存
void read_file(const char *filename, unsigned char **data, size_t *size)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);
    *data = (unsigned char *)malloc(*size);
    fread(*data, 1, *size, file);
    fclose(file);
}

// 保存内存数据到文件
void write_file(const char *filename, const unsigned char *data, size_t size)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fwrite(data, 1, size, file);
    fclose(file);
}

// 加载 OpenCL kernel 文件
char *load_kernel_source(const char *filename, size_t *size)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Failed to open kernel file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *source = (char *)malloc(*size + 1);
    fread(source, 1, *size, file);
    source[*size] = '\0';
    fclose(file);
    return source;
}

int main()
{
    // 1. 读取 YUV 或 RGB 文件
    const char *input_filename = "input.yuv";         // 输入文件
    const char *output_filename = "output.rgb";       // 输出文件
    const char *output_resize_filename = "scale.rgb"; // 输出文件
    unsigned char *input_data;
    size_t input_size;
    read_file(input_filename, &input_data, &input_size);

    // 假设输入是 NV21 格式的 YUV 图像
    int width = 1280;               // 图像宽度
    int height = 720;               // 图像高度
    size_t y_size = width * height; // Y 分量大小
    size_t uv_size = y_size / 2;    // UV 分量大小

    // 2. 初始化 OpenCL
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_program program_resize;
    cl_kernel kernel_resize;
    cl_int err;

    // 获取平台和设备
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err, "Failed to get platform ID");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err, "Failed to get device ID");

    // 创建上下文和命令队列
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err, "Failed to create context");

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err, "Failed to create command queue");

    char options[128];
    sprintf(options, "-D PIX_PER_WI_Y=1 -D SCN=1 -D DCN=3 -D BIDX=2 -D UIDX=1 -D SRC_DEPTH=0");

    // 3. 加载 kernel 文件
    size_t kernel_size;
    char *kernel_source = load_kernel_source("color_yuv.cl", &kernel_size);

    // 创建和编译程序
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, &kernel_size, &err);
    CHECK_ERROR(err, "Failed to create program");

    err = clBuildProgram(program, 1, &device, options, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    // 4. 创建内核
    kernel = clCreateKernel(program, "YUV2RGB_NVx", &err); // 使用 YUV 转 RGB 的 kernel
    CHECK_ERROR(err, "Failed to create kernel");

    // 3. 加载 kernel 文件
    size_t kernel_resize_size;
    char *kernel_resize_source = load_kernel_source("resize.cl", &kernel_resize_size);

    // 创建和编译程序
    program_resize = clCreateProgramWithSource(context, 1, (const char **)&kernel_resize_source, &kernel_resize_size, &err);
    CHECK_ERROR(err, "Failed to create program 2");
    char options2[128];
    sprintf(options2, "-D SRC_DEPTH=0 -D INTER_LINEAR_INTEGER -D T=uchar  -D WT=int3 -D CONVERT_TO_WT=convert_int3 -D CONVERT_TO_DT=convert_uchar -D CN=3 -D T1=uchar ");
    err = clBuildProgram(program_resize, 1, &device, options2, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t log_size;
        clGetProgramBuildInfo(program_resize, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program_resize, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build log 2:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    // 4. 创建内核
    kernel_resize = clCreateKernel(program_resize, "resizeLN", &err);
    CHECK_ERROR(err, "Failed to create kernel 2");

    // 设置kernel参数
    int rows = height;       // 假设图像高度为1080
    int cols = width;        // 假设图像宽度为1920
    int src_step = cols;     // YUV图像的步长
    int dst_step = cols * 3; // RGB图像的步长（假设为3通道）
    int src_offset = 0;      // YUV图像的偏移量
    int dt_offset = 0;       // RGB图像的偏移量
    int PIX_PER_WI_Y = 1;    // 每个工作项处理的像素行数

    // 分配内存
    size_t src_size = rows * src_step * 3 / 2;
    size_t dst_size = rows * dst_step;
    unsigned char *rgb_data = (unsigned char *)malloc(dst_size);

    cl_mem src_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, src_size, input_data, &err);
    CHECK_ERROR(err, "Failed to create source buffer");

    cl_mem dst_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size, NULL, &err);
    CHECK_ERROR(err, "Failed to create destination buffer");

    // 设置kernel参数
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_buffer);
    CHECK_ERROR(err, "Failed to set kernel argument 0");

    err = clSetKernelArg(kernel, 1, sizeof(int), &src_step);
    CHECK_ERROR(err, "Failed to set kernel argument 1");

    err = clSetKernelArg(kernel, 2, sizeof(int), &src_offset);
    CHECK_ERROR(err, "Failed to set kernel argument 2");

    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &dst_buffer);
    CHECK_ERROR(err, "Failed to set kernel argument 3");

    err = clSetKernelArg(kernel, 4, sizeof(int), &dst_step);
    CHECK_ERROR(err, "Failed to set kernel argument 4");

    err = clSetKernelArg(kernel, 5, sizeof(int), &dt_offset);
    CHECK_ERROR(err, "Failed to set kernel argument 5");

    err = clSetKernelArg(kernel, 6, sizeof(int), &rows);
    CHECK_ERROR(err, "Failed to set kernel argument 6");

    err = clSetKernelArg(kernel, 7, sizeof(int), &cols);
    CHECK_ERROR(err, "Failed to set kernel argument 7");
    cl_event kernel_event;
    // 执行kernel
    size_t global_work_size[2] = {cols / 2, rows / 2 / PIX_PER_WI_Y};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, &kernel_event);
    CHECK_ERROR(err, "Failed to enqueue kernel");

    // 等待 Kernel 完成
    clWaitForEvents(1, &kernel_event);

    // 获取时间戳
    cl_ulong start_time, end_time;
    err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    CHECK_ERROR(err, "Failed to get start time");

    err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
    CHECK_ERROR(err, "Failed to get end time");

    // 计算耗时
    cl_ulong elapsed_time = end_time - start_time; // 纳秒
    double elapsed_time_ms = elapsed_time / 1e6;   // 毫秒
    double elapsed_time_s = elapsed_time / 1e9;    // 秒

    printf("Kernel execution time: %f ms\n", elapsed_time_ms);

    // 8. 读取结果
    err = clEnqueueReadBuffer(queue, dst_buffer, CL_TRUE, 0, dst_size, rgb_data, 0, NULL, NULL);
    CHECK_ERROR(err, "Failed to read RGB buffer");

    // 9. 保存结果
    write_file(output_filename, rgb_data, dst_size);

    // 4. 准备输入数
    const int INTER_RESIZE_COEF_SCALE = 2048; // Example value, adjust as needed
    int src_rows = height;                    // 输入图像的行数
    int src_cols = width;                     // 输入图像的列数
    int src_step_resize = src_cols * 3;       // 输入图像的列数
    int dst_rows = 480;                      // 输出图像的行数
    int dst_cols = 640;                      // 输出图像的列数
    int dst_step_resize = dst_cols * 3;
    int scale_size = dst_rows * dst_step_resize;
    unsigned char *rgb_data_resize = (unsigned char *)malloc(scale_size);
    const float inv_fx = (float)src_cols / dst_cols; // 输入宽度 / 输出宽度
    const float inv_fy = (float)src_rows / dst_rows; // 输入高度 / 输出高度

    // Calculate buffer size
    size_t buffer_size = (dst_cols + dst_rows) * sizeof(int) + (dst_cols + dst_rows) * 2 * sizeof(short);
    unsigned char *buffer_data = (unsigned char *)malloc(buffer_size);

    // Cast pointers to the appropriate types
    int *xofs = (int *)buffer_data;
    int *yofs = xofs + dst_cols;
    short *ialpha = (short *)(yofs + dst_rows);
    short *ibeta = ialpha + dst_cols * 2;

    // Calculate xofs and ialpha
    for (int dx = 0; dx < dst_cols; dx++)
    {
        float fxx = (float)((dx + 0.5) * inv_fx - 0.5);
        int sx = (int)floor(fxx);
        fxx -= sx;

        if (sx < 0)
        {
            fxx = 0, sx = 0;
        }
        if (sx >= src_cols - 1)
        {
            fxx = 0, sx = src_cols - 1;
        }

        xofs[dx] = sx;
        ialpha[dx * 2 + 0] = (short)((1.f - fxx) * INTER_RESIZE_COEF_SCALE);
        ialpha[dx * 2 + 1] = (short)(fxx * INTER_RESIZE_COEF_SCALE);
    }

    // Calculate yofs and ibeta
    for (int dy = 0; dy < dst_rows; dy++)
    {
        float fyy = (float)((dy + 0.5) * inv_fy - 0.5);
        int sy = (int)floor(fyy);
        fyy -= sy;

        yofs[dy] = sy;
        ibeta[dy * 2 + 0] = (short)((1.f - fyy) * INTER_RESIZE_COEF_SCALE);
        ibeta[dy * 2 + 1] = (short)(fyy * INTER_RESIZE_COEF_SCALE);
    }

    cl_mem src_buffer_resize = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dst_size, rgb_data, &err);
    CHECK_ERROR(err, "Failed to create source buffer");

    cl_mem dst_buffer_resize = clCreateBuffer(context, CL_MEM_WRITE_ONLY, scale_size, NULL, &err);
    CHECK_ERROR(err, "Failed to create destination buffer");

    cl_mem buffer_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_size, buffer_data, &err);
    CHECK_ERROR(err, "Failed to create coefficients buffer");

    int default_z = 0;

    // 6. 设置 kernel 参数
    // Set kernel arguments
    err = clSetKernelArg(kernel_resize, 0, sizeof(cl_mem), &src_buffer_resize);
    CHECK_ERROR(err, "Failed to set kernel_resize argument 0");

    err = clSetKernelArg(kernel_resize, 1, sizeof(int), &src_step_resize);
    CHECK_ERROR(err, "Failed to set kernel_resize argument 1");

    err = clSetKernelArg(kernel_resize, 2, sizeof(int), &src_offset);
    CHECK_ERROR(err, "Failed to set kernel_resize argument 2");

    err = clSetKernelArg(kernel_resize, 3, sizeof(int), &src_rows);
    CHECK_ERROR(err, "Failed to set kernel_resize argument 3");

    err = clSetKernelArg(kernel_resize, 4, sizeof(int), &src_cols);
    CHECK_ERROR(err, "Failed to set kernel_resize argument 4");

    err = clSetKernelArg(kernel_resize, 5, sizeof(cl_mem), &dst_buffer_resize);
    CHECK_ERROR(err, "Failed to set kernel_resize argument 5");

    err = clSetKernelArg(kernel_resize, 6, sizeof(int), &dst_step_resize);
    CHECK_ERROR(err, "Failed to set kernel_resize argument 6");

    err = clSetKernelArg(kernel_resize, 7, sizeof(int), &dt_offset);
    CHECK_ERROR(err, "Failed to set kernel_resize argument 7");

    err = clSetKernelArg(kernel_resize, 8, sizeof(int), &dst_rows);
    CHECK_ERROR(err, "Failed to set kernel_resize argument 8");

    err = clSetKernelArg(kernel_resize, 9, sizeof(int), &dst_cols);
    CHECK_ERROR(err, "Failed to set kernel_resize argument 9");

    err = clSetKernelArg(kernel_resize, 10, sizeof(cl_mem), &buffer_mem);
    CHECK_ERROR(err, "Failed to set kernel_resize argument 10");

    cl_event kernel_event_resize;
    // 执行kernel
    size_t global_work_size_resize[2] = {dst_cols, dst_rows};
    err = clEnqueueNDRangeKernel(queue, kernel_resize, 2, NULL, global_work_size_resize, NULL, 0, NULL, &kernel_event_resize);
    CHECK_ERROR(err, "Failed to enqueue kernel");

    // 等待 Kernel 完成
    clWaitForEvents(1, &kernel_event_resize);

    // 获取时间戳
    // cl_ulong start_time, end_time;
    err = clGetEventProfilingInfo(kernel_event_resize, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    CHECK_ERROR(err, "Failed to get start time");

    err = clGetEventProfilingInfo(kernel_event_resize, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
    CHECK_ERROR(err, "Failed to get end time");

    // 计算耗时
    elapsed_time = end_time - start_time; // 纳秒
    elapsed_time_ms = elapsed_time / 1e6; // 毫秒

    printf("Kernel execution time: %f ms\n", elapsed_time_ms);

    // 8. 读取结果
    err = clEnqueueReadBuffer(queue, dst_buffer_resize, CL_TRUE, 0, scale_size, rgb_data_resize, 0, NULL, NULL);
    CHECK_ERROR(err, "Failed to read RGB buffer");

    // 9. 保存结果
    write_file(output_resize_filename, rgb_data_resize, scale_size);

    // 10. 清理资源
    clReleaseMemObject(src_buffer);
    clReleaseMemObject(dst_buffer);
    clReleaseMemObject(src_buffer_resize);
    clReleaseMemObject(dst_buffer_resize);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(input_data);
    free(rgb_data);
    free(kernel_source);

    printf("Processing complete!\n");
    return 0;
}
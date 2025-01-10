// #define CL_TARGET_OPENCL_VERSION 120 // 使用 OpenCL 1.2
#include <stdio.h> // 使用 printf
#include <vector>
#include <CL/cl.h>
#include <fstream>

// 检查 OpenCL 错误
#define CHECK_CL_ERROR(err)                                     \
    if (err != CL_SUCCESS)                                      \
    {                                                           \
        printf("OpenCL error: %d at line %d\n", err, __LINE__); \
        exit(1);                                                \
    }

// 读取文件内容
std::string readFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        printf("Failed to open file: %s\n", filename.c_str());
        exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

int main()
{
    // 初始化 OpenCL
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program_image2d, program_manual;
    cl_kernel kernel_image2d, kernel_manual;
    cl_int err;

    // 获取平台和设备
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_CL_ERROR(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_CL_ERROR(err);

    // 创建上下文和命令队列
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL_ERROR(err);

    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    CHECK_CL_ERROR(err);

    // 读取内核文件
    std::string source_image2d = readFile("resize_rgb_bilinear_image2d.cl");
    std::string source_manual = readFile("resize_rgb_bilinear.cl");

    // 创建程序对象
    const char *source_image2d_ptr = source_image2d.c_str();
    program_image2d = clCreateProgramWithSource(context, 1, &source_image2d_ptr, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clBuildProgram(program_image2d, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        char buildLog[16384];
        clGetProgramBuildInfo(program_image2d, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        printf("Build log (image2d): %s\n", buildLog);
        exit(1);
    }

    const char *source_manual_ptr = source_manual.c_str();
    program_manual = clCreateProgramWithSource(context, 1, &source_manual_ptr, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clBuildProgram(program_manual, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        char buildLog[16384];
        clGetProgramBuildInfo(program_manual, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        printf("Build log (manual): %s\n", buildLog);
        exit(1);
    }

    // 创建内核对象
    kernel_image2d = clCreateKernel(program_image2d, "resize_rgb_bilinear", &err);
    CHECK_CL_ERROR(err);
    kernel_manual = clCreateKernel(program_manual, "resize_rgb_bilinear", &err);
    CHECK_CL_ERROR(err);

    // 输入和输出图像尺寸
    int input_width = 7680;
    int input_height = 1300;
    int output_width = 960;
    int output_height = 540;

    // 分配输入和输出数据
    std::vector<unsigned char> input_data(input_width * input_height * 3);
    std::vector<unsigned char> output_data(output_width * output_height * 3);

    // 填充输入数据（示例：随机数据）
    for (size_t i = 0; i < input_data.size(); i++)
    {
        input_data[i] = rand() % 256;
    }

    // 创建 OpenCL 图像和缓冲区
    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc desc = {};
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = input_width;
    desc.image_height = input_height;
    desc.image_row_pitch = 0;   // 设置为 0 表示连续内存
    desc.image_slice_pitch = 0; // 设置为 0 表示连续内存

    // 将 RGB 数据转换为 RGBA
    std::vector<unsigned char> rgba_data(input_width * input_height * 4);
    for (int i = 0; i < input_width * input_height; i++)
    {
        rgba_data[i * 4 + 0] = input_data[i * 3 + 0]; // R
        rgba_data[i * 4 + 1] = input_data[i * 3 + 1]; // G
        rgba_data[i * 4 + 2] = input_data[i * 3 + 2]; // B
        rgba_data[i * 4 + 3] = 255;                   // A (不透明度)
    }

    cl_mem input_image = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, &desc, rgba_data.data(), &err);
    CHECK_CL_ERROR(err);
    cl_mem output_image = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, NULL, &err);
    CHECK_CL_ERROR(err);

    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_width * input_height * 3, input_data.data(), &err);
    CHECK_CL_ERROR(err);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_width * output_height * 3, NULL, &err);
    CHECK_CL_ERROR(err);

    // 设置内核参数
    float scale_x = (float)input_width / output_width;
    float scale_y = (float)input_height / output_height;
    err = clSetKernelArg(kernel_image2d, 0, sizeof(cl_mem), &input_image);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel_image2d, 1, sizeof(cl_mem), &output_image);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel_image2d, 2, sizeof(float), &scale_x);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel_image2d, 3, sizeof(float), &scale_y);
    CHECK_CL_ERROR(err);

    err = clSetKernelArg(kernel_manual, 0, sizeof(cl_mem), &input_buffer);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel_manual, 1, sizeof(cl_mem), &output_buffer);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel_manual, 2, sizeof(int), &input_width);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel_manual, 3, sizeof(int), &input_height);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel_manual, 4, sizeof(int), &output_width);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel_manual, 5, sizeof(int), &output_height);
    CHECK_CL_ERROR(err);
    // 运行内核并测量时间
    size_t global_size[2] = {static_cast<size_t>(output_width), static_cast<size_t>(output_height)};
    cl_event event;
    double total_time_image2d = 0.0;
    double total_time_manual = 0.0;
    int num_runs = 1000;

    for (int i = 0; i < num_runs; i++)
    {
        // 运行 image2d_t 实现
        err = clEnqueueNDRangeKernel(queue, kernel_image2d, 2, NULL, global_size, NULL, 0, NULL, &event);
        CHECK_CL_ERROR(err);
        clWaitForEvents(1, &event);

        cl_ulong start, end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
        total_time_image2d += (end - start) * 1e-6; // 转换为毫秒

        // 运行手动实现
        err = clEnqueueNDRangeKernel(queue, kernel_manual, 2, NULL, global_size, NULL, 0, NULL, &event);
        CHECK_CL_ERROR(err);
        clWaitForEvents(1, &event);

        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
        total_time_manual += (end - start) * 1e-6; // 转换为毫秒
    }

    // 输出结果
    printf("Average time (image2d_t): %.3f ms\n", total_time_image2d / num_runs);
    printf("Average time (manual): %.3f ms\n", total_time_manual / num_runs);

    // 暂停程序
    printf("Press Enter to exit...\n");
    getchar(); // 等待用户输入

    // 释放资源
    clReleaseMemObject(input_image);
    clReleaseMemObject(output_image);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel_image2d);
    clReleaseKernel(kernel_manual);
    clReleaseProgram(program_image2d);
    clReleaseProgram(program_manual);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
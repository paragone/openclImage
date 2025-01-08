#pragma once

#include <memory>
#include <CL/cl.h>
#include <variant>
#include <iostream>
#include <unordered_map>
#include <sys/mman.h>
#include <stdexcept>
#include "BufferPool.h"

namespace bos::mm
{
    class Plane
    {
    public:
        Plane(size_t width, size_t height, size_t stride, std::shared_ptr<Buffer> buffer)
            : mBuffer(std::move(buffer)), mWidth(width), mHeight(height), mStride(stride) {}

        // 获取图像平面的宽度
        size_t get_width() const { return mWidth; }

        // 获取图像平面的高度
        size_t get_height() const { return mHeight; }

        // 获取图像平面的步幅（stride）
        size_t get_stride() const { return mStride; }

        // 获取数据指针
        uint8_t *get_data() { return mBuffer->get_data(); }

        // 转换为 OpenCL 缓冲区
        cl_mem to_cl_mem(cl_context context, cl_command_queue queue, bool is_dma = false)
        {
            if (is_dma && mBuffer->get_type() == Buffer::Type::DMABUF)
            {
                // 使用 DMA FD 创建 cl_mem
                int fd = mBuffer->get_dma_fd();
                cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
                return clCreateBufferFromFd(context, fd, flags, get_stride() * get_height(), nullptr);
            }
            else
            {
                // 使用普通内存创建 cl_mem
                uint8_t *ptr = get_data();
                if (ptr == nullptr)
                {
                    std::cerr << "Failed to get plane data." << std::endl;
                    return nullptr;
                }
                cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
                return clCreateBuffer(context, flags, get_stride() * get_height(), ptr, nullptr);
            }
        }

    private:
        std::shared_ptr<Buffer> mBuffer; // 使用智能指针来避免数据拷贝
        size_t mWidth, mHeight, mStride; // 图像平面特定的属性
    };
}

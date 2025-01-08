#pragma once
#include <memory>
#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <unordered_map>

#include "BufferPool.h"
#include "Plane.h"

namespace bos::mm
{
    class Image
    {
    public:
        enum class Format
        {
            RGB,
            RGBA,
            YUV420,
            YUV422,
            NV21,
            NV12,
            // 更多格式
        };

        // 从 BufferPool 获取缓冲区自动构造
        Image(Format format, size_t width, size_t height, BufferPool &buffer_pool)
            : format(format), width(width), height(height), buffer_pool(buffer_pool)
        {
            create_planes_from_pool();
        }

        // 从外部 Buffer 创建
        Image(Format format, size_t width, size_t height, std::vector<std::shared_ptr<Buffer>> external_buffers)
            : format(format), width(width), height(height), external_buffers(std::move(external_buffers))
        {
            create_planes_from_external_buffers();
        }

        // 创建多个 Plane，自动从 BufferPool 获取
        void create_planes_from_pool()
        {
            // 根据格式生成相应的 Plane
            if (format == Format::RGB || format == Format::RGBA)
            {
                // 使用 BufferPool 获取 Buffer 创建 Plane
                auto buffer = buffer_pool.get_buffer(Buffer::Type::NORMAL, width * height * 3);
                planes.push_back(std::make_shared<Plane>(width, height, width * 3, buffer));
            }
            else if (format == Format::NV21 || format == Format::NV12)
            {
                // 获取两个缓冲区，Y 和 UV 分别创建 Plane
                auto buffer_y = buffer_pool.get_buffer(Buffer::Type::NORMAL, width * height);
                planes.push_back(std::make_shared<Plane>(width, height, width, buffer_y));
                auto buffer_uv = buffer_pool.get_buffer(Buffer::Type::NORMAL, width * height / 2);
                planes.push_back(std::make_shared<Plane>(width / 2, height / 2, width / 2, buffer_uv));
            }
            else if (format == Format::YUV420 || format == Format::YUV422)
            {
                // 获取三个缓冲区，Y、U、V 分别创建 Plane
                auto buffer_y = buffer_pool.get_buffer(Buffer::Type::NORMAL, width * height);
                planes.push_back(std::make_shared<Plane>(width, height, width, buffer_y));
                auto buffer_u = buffer_pool.get_buffer(Buffer::Type::NORMAL, width * height / 4);
                planes.push_back(std::make_shared<Plane>(width / 2, height / 2, width / 2, buffer_u));
                auto buffer_v = buffer_pool.get_buffer(Buffer::Type::NORMAL, width * height / 4);
                planes.push_back(std::make_shared<Plane>(width / 2, height / 2, width / 2, buffer_v));
            }
        }

        // 从外部 Buffer 创建多个 Plane
        void create_planes_from_external_buffers()
        {
            // 根据格式生成相应的 Plane
            if (format == Format::RGB || format == Format::RGBA)
            {
                // 每个 Plane 使用外部的 Buffer
                planes.push_back(std::make_shared<Plane>(width, height, width * 3, external_buffers[0]));
            }
            else if (format == Format::NV21 || format == Format::NV12)
            {
                // 使用外部提供的缓冲区创建 Plane
                planes.push_back(std::make_shared<Plane>(width, height, width, external_buffers[0]));
                planes.push_back(std::make_shared<Plane>(width / 2, height / 2, width / 2, external_buffers[1]));
            }
            else if (format == Format::YUV420 || format == Format::YUV422)
            {
                planes.push_back(std::make_shared<Plane>(width, height, width, *external_buffers[0]));
                planes.push_back(std::make_shared<Plane>(width / 2, height / 2, width / 2, external_buffers[1]));
                planes.push_back(std::make_shared<Plane>(width / 2, height / 2, width / 2, external_buffers[2]));
            }
        }

        // 获取所有 Plane 的引用
        const std::vector<std::shared_ptr<Plane>> &get_planes() const
        {
            return planes;
        }

    private:
        Format format;
        size_t width, height;
        BufferPool &buffer_pool;                               // 引用 BufferPool 用于从中获取缓冲区
        std::vector<std::shared_ptr<Buffer>> external_buffers; // 外部传入的缓冲区
        std::vector<std::shared_ptr<Plane>> planes;            // 存储多个 Plane
    };
}

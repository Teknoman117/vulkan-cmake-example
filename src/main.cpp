#define GLFW_INCLUDE_VULKAN
#define VULKAN_HPP_TYPESAFE_CONVERSION
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <memory>
#include <set>
#include <vector>

extern "C"
{
    PFN_vkCreateDebugReportCallbackEXT createDebugReportCallbackEXT;
    PFN_vkDestroyDebugReportCallbackEXT destroyDebugReportCallbackEXT;

    VkResult vkCreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo,
            const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback)
    {
        if (!createDebugReportCallbackEXT)
            return VK_ERROR_EXTENSION_NOT_PRESENT;

        return createDebugReportCallbackEXT(instance, pCreateInfo, pAllocator, pCallback);
    }

    void vkDestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback,
            const VkAllocationCallbacks* pAllocator)
    {
        if (!destroyDebugReportCallbackEXT)
            return;

        return destroyDebugReportCallbackEXT(instance, callback, pAllocator);
    }
}

namespace
{
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugReportFlagsEXT flags,
        VkDebugReportObjectTypeEXT objType,
        uint64_t obj,
        size_t location,
        int32_t code,
        const char* layerPrefix,
        const char* msg,
        void* userData)
    {

        std::cerr << "[DEBUG] [VALIDATION LAYER MESSAGE]: " << msg << std::endl;
        return VK_FALSE;
    }

    static std::vector<char> readFile(const std::string& path)
    {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("failed to open file!");

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        return buffer;
    }
}

struct QueueFamilyIndices
{
    uint32_t graphicsFamily = std::numeric_limits<uint32_t>::max();
    uint32_t presentFamily = std::numeric_limits<uint32_t>::max();

    constexpr bool isComplete() {
        return graphicsFamily != std::numeric_limits<uint32_t>::max()
                && presentFamily != std::numeric_limits<uint32_t>::max();
    }
};

struct SwapchainSupportDetails
{
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

class HelloTriangleApplication
{
    constexpr static uint32_t kMaxInFlight = 2;
    constexpr static uint32_t kWidth = 800;
    constexpr static uint32_t kHeight = 600;

#ifdef NDEBUG
    constexpr static bool kValidationEnabled = false;
#else
    constexpr static bool kValidationEnabled = true;
#endif

    // collect list of desired layers
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_LUNARG_standard_validation"
    };

    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    GLFWwindow* mWindow;
    vk::UniqueInstance mInstance;
    vk::UniqueDebugReportCallbackEXT mDebugReportCallback;
    vk::PhysicalDevice mPhysicalDevice;
    vk::UniqueDevice mDevice;
    vk::Queue mGraphicsQueue;
    vk::Queue mPresentQueue;

    vk::UniqueRenderPass mRenderPass;
    vk::UniqueShaderModule mVertexShaderModule;
    vk::UniqueShaderModule mFragmentShaderModule;
    vk::UniquePipelineLayout mPipelineLayout;
    vk::UniquePipeline mPipeline;
    std::vector<vk::UniqueSemaphore> mImageAvailableSemaphores;
    std::vector<vk::UniqueSemaphore> mRenderFinishedSemaphores;
    std::vector<vk::UniqueFence> mInFlightFences;

    vk::UniqueSurfaceKHR mSurface;
    vk::UniqueSwapchainKHR mSwapchain;
    vk::Format mSwapchainImageFormat;
    vk::Extent2D mSwapchainExtent;
    std::vector<vk::Image> mSwapchainImages;
    std::vector<vk::UniqueImageView> mSwapchainImageViews;
    std::vector<vk::UniqueFramebuffer> mSwapchainFramebuffers;
    vk::UniqueCommandPool mCommandPool;
    std::vector<vk::CommandBuffer> mCommandBuffers;

    size_t currentFrame = 0;

public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        mWindow = glfwCreateWindow(kWidth, kHeight, "Vulkan", nullptr, nullptr);
    }

    void initVulkan()
    {
        createInstance();
        createSurface();
        createDevice();
        createSwapchain();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
        createSemaphores();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(mWindow))
        {
            glfwPollEvents();
            drawFrame();
        }

        mDevice->waitIdle();
    }

    void cleanup()
    {
        glfwDestroyWindow(mWindow);
        glfwTerminate();
    }

    /**
     * Check if desired validation layers are supported
     * 
     * @return true if validation support is available
     */
    bool checkValidationLayerSupport()
    {
        auto availableLayers = vk::enumerateInstanceLayerProperties();
        for (auto& layer : validationLayers)
        {
            auto predicate = [layer] (auto& availableLayer) {
                return std::string_view(availableLayer.layerName) == std::string_view(layer);
            };

            if (std::find_if(availableLayers.begin(), availableLayers.end(), predicate) == availableLayers.end())
                return false;
        }
        return true;
    }

    std::vector<const char*> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (kValidationEnabled) {
            extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }

        return extensions;
    }

    void createInstance()
    {
        // check validation layer support
        if (kValidationEnabled && !checkValidationLayerSupport())
            throw std::runtime_error("validation layers requrested, but not available!");

        vk::ApplicationInfo appInfo("Hello Triangle", VK_MAKE_VERSION(0, 1, 0), "No Engine", VK_MAKE_VERSION(0, 1, 0),
                VK_API_VERSION_1_1);

        auto extensions = getRequiredExtensions();
        mInstance = vk::createInstanceUnique({{}, &appInfo, kValidationEnabled ? (uint32_t) validationLayers.size() : 0,
                kValidationEnabled ? validationLayers.data() : nullptr, (uint32_t) extensions.size(),
                extensions.data()});
        if (!mInstance)
            throw std::runtime_error("failed to create instance!");

        // print extensions used by instance
        std::cerr << "---- Instance Extensions ----" << std::endl;
        for (auto& extension : vk::enumerateInstanceExtensionProperties())
        {
            std::cerr << "Found: " << extension.extensionName << std::endl;
        }
        std::cerr << "-----------------------------" << std::endl;

        if (kValidationEnabled)
            setupDebugCallback();
    }

    void setupDebugCallback()
    {
        // load extension functions
        createDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)
            mInstance->getProcAddr("vkCreateDebugReportCallbackEXT");
        destroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)
            mInstance->getProcAddr("vkDestroyDebugReportCallbackEXT");

        if (!createDebugReportCallbackEXT || !destroyDebugReportCallbackEXT)
        {
            throw std::runtime_error("failed to load extensions");
        }

        // register debug callback
        mDebugReportCallback = mInstance->createDebugReportCallbackEXTUnique({vk::DebugReportFlagBitsEXT::eError |
                vk::DebugReportFlagBitsEXT::eWarning, &debugCallback});
        if (!mDebugReportCallback)
            throw std::runtime_error("failed to create debug report callback!");
    }

    void createSurface()
    {
        vk::SurfaceKHR surface;
        vk::Result result = static_cast<vk::Result>(glfwCreateWindowSurface(mInstance.get(), mWindow, nullptr,
                reinterpret_cast<VkSurfaceKHR*>(&surface)));

        vk::ObjectDestroy<vk::Instance> deleter(mInstance.get(), nullptr);
        mSurface = vk::createResultValue(result, surface, "createSurface", deleter);
    }

    QueueFamilyIndices getQueueFamilyIndices(vk::PhysicalDevice& d)
    {
        QueueFamilyIndices indices;

        auto queueFamilies = d.getQueueFamilyProperties();
        for (auto it = queueFamilies.begin(); it != queueFamilies.end(); it++)
        {
            uint32_t id = std::distance(queueFamilies.begin(), it);
            if (it->queueFlags & vk::QueueFlagBits::eGraphics)
                indices.graphicsFamily = id;

            vk::Bool32 supported;
            d.getSurfaceSupportKHR(id, mSurface.get(), &supported);
            if (supported)
                indices.presentFamily = id;
        }
        return indices;
    }

    void createDevice()
    {
        auto devices = mInstance->enumeratePhysicalDevices();

        // Get the first device that supports required features
        for (auto& device : devices)
        {
            mPhysicalDevice = device;
            std::cerr << "[INFO] Attempting Device: " << device.getProperties().deviceName << std::endl;

            auto details = getSwapchainSupportDetails(device);
            if (details.formats.empty() || details.presentModes.empty())
                continue;

            // Assemble list of UNIQUE queue families to be created
            QueueFamilyIndices indices = getQueueFamilyIndices(device);
            const std::set<uint32_t> queueFamilies = { indices.graphicsFamily, indices.presentFamily };

            // Transform queue families into queue create infos
            float defaultPriority = 1.0f;
            std::vector<vk::DeviceQueueCreateInfo> queues;
            std::transform(queueFamilies.begin(), queueFamilies.end(), std::back_inserter(queues),
                    [indices, &defaultPriority] (auto& family) -> vk::DeviceQueueCreateInfo {
                return { {}, indices.graphicsFamily, 1, &defaultPriority };
            });

            // note: device layers have been deprecated
            mDevice = device.createDeviceUnique({{}, (uint32_t) queues.size(), queues.data(), 0U, nullptr,
                    (uint32_t) deviceExtensions.size(), deviceExtensions.data()});
            if (!mDevice)
                continue;

            mGraphicsQueue = mDevice->getQueue(indices.graphicsFamily, 0);
            if (!mGraphicsQueue)
                throw std::runtime_error("failed to get graphics queue!");

            mPresentQueue = mDevice->getQueue(indices.presentFamily, 0);
            if (!mPresentQueue)
                throw std::runtime_error("failed to get present queue!");

            return;
        }
        throw std::runtime_error("failed to create logical device!");
    }

    SwapchainSupportDetails getSwapchainSupportDetails(vk::PhysicalDevice& d)
    {
        return (SwapchainSupportDetails) {
            d.getSurfaceCapabilitiesKHR(mSurface.get()),
            d.getSurfaceFormatsKHR(mSurface.get()),
            d.getSurfacePresentModesKHR(mSurface.get())
        };
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats)
    {
        if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined)
            return { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };

        for (const auto& availableFormat : formats)
        {
            if (availableFormat.format == vk::Format::eB8G8R8A8Unorm
                    && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                return availableFormat;
        }
        return formats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> presentModes)
    {
        vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;
        for (const auto& presentMode : presentModes)
        {
            if (presentMode == vk::PresentModeKHR::eMailbox)
                return presentMode;

            else if (presentMode == vk::PresentModeKHR::eImmediate)
                bestMode = presentMode;
        }
        return bestMode;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return capabilities.currentExtent;

        return {
            std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, kWidth)),
            std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, kHeight))
        };
    }

    void createSwapchain()
    {
        auto swapchainSupportDetails = getSwapchainSupportDetails(mPhysicalDevice);
        auto surfaceFormat = chooseSwapSurfaceFormat(swapchainSupportDetails.formats);
        auto presentMode = chooseSwapPresentMode(swapchainSupportDetails.presentModes);
        auto extent = chooseSwapExtent(swapchainSupportDetails.capabilities);

        uint32_t imageCount = swapchainSupportDetails.capabilities.minImageCount + 1;
        if (swapchainSupportDetails.capabilities.maxImageCount > 0
                && imageCount > swapchainSupportDetails.capabilities.maxImageCount)
        {
            imageCount = swapchainSupportDetails.capabilities.maxImageCount;
        }

        std::cerr << "[INFO] Using " << imageCount << " images in swapchain" << std::endl;

        auto indices = getQueueFamilyIndices(mPhysicalDevice);
        const std::vector<uint32_t> queueFamilyIndices = { indices.graphicsFamily, indices.presentFamily };
    
        mSwapchain = mDevice->createSwapchainKHRUnique({
            {},
            mSurface.get(),
            imageCount,
            surfaceFormat.format,
            surfaceFormat.colorSpace,
            extent,
            1,
            vk::ImageUsageFlagBits::eColorAttachment,
            (indices.graphicsFamily != indices.presentFamily) ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
            (indices.graphicsFamily != indices.presentFamily) ? (uint32_t) queueFamilyIndices.size() : 0,
            (indices.graphicsFamily != indices.presentFamily) ? queueFamilyIndices.data() : nullptr,
            swapchainSupportDetails.capabilities.currentTransform,
            vk::CompositeAlphaFlagBitsKHR::eOpaque,
            presentMode,
            true
        });
        if (!mSwapchain)
            throw std::runtime_error("failed to create swapchain!");

        mSwapchainImageFormat = surfaceFormat.format;
        mSwapchainExtent = extent;

        // create image views from all swapchain imagesc
        mSwapchainImages = mDevice->getSwapchainImagesKHR(mSwapchain.get());
        std::transform(mSwapchainImages.begin(), mSwapchainImages.end(), std::back_inserter(mSwapchainImageViews),
                [this] (auto& image) {
            return mDevice->createImageViewUnique({{}, image, vk::ImageViewType::e2D, mSwapchainImageFormat, {},
                    {vk::ImageAspectFlagBits::eColor, 0U, 1U, 0U, 1U}});
        });
    }

    void createRenderPass()
    {
        vk::AttachmentDescription colorAttachment = {{}, mSwapchainImageFormat, vk::SampleCountFlagBits::e1,
                vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
                vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
                vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR};

        vk::AttachmentReference colorAttachmentRef = {0U, vk::ImageLayout::eColorAttachmentOptimal};
        vk::SubpassDescription subpass = {{}, vk::PipelineBindPoint::eGraphics, 0U, nullptr, 1U, &colorAttachmentRef};

        vk::SubpassDependency dependency = { VK_SUBPASS_EXTERNAL, 0U, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                vk::PipelineStageFlagBits::eColorAttachmentOutput, {},
                vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite};

        mRenderPass = mDevice->createRenderPassUnique({{}, 1U, &colorAttachment, 1U, &subpass, 1U, &dependency});
        if (!mRenderPass)
            throw std::runtime_error("failed to create render pass!");
    }

    vk::UniqueShaderModule createShaderModule(const std::string& path)
    {
        auto object = readFile(path);
        return mDevice->createShaderModuleUnique({{}, object.size(), reinterpret_cast<const uint32_t*>(object.data())});
    }

    void createGraphicsPipeline()
    {
        if (!((mVertexShaderModule = createShaderModule("shaders/triangle/vert.spv")))
                || !((mFragmentShaderModule = createShaderModule("shaders/triangle/frag.spv"))))
        {
            throw std::runtime_error("failed to shader create modules!");
        }

        vk::PipelineShaderStageCreateInfo shaderStages[] = {
            {{}, vk::ShaderStageFlagBits::eVertex, mVertexShaderModule.get(), "main"},
            {{}, vk::ShaderStageFlagBits::eFragment, mFragmentShaderModule.get(), "main"}
        };

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {{}, 0U, nullptr, 0U, nullptr};
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {{}, vk::PrimitiveTopology::eTriangleList, false};
        vk::Viewport viewport = {0.0f, 0.0f, (float) mSwapchainExtent.width, (float) mSwapchainExtent.height, 0.0f, 1.0f};
        vk::Rect2D scissor = {{0, 0}, mSwapchainExtent};
        vk::PipelineViewportStateCreateInfo viewportState = {{}, 1U, &viewport, 1U, &scissor};
        vk::PipelineRasterizationStateCreateInfo rasterizer = {{}, false, false, vk::PolygonMode::eFill,
                vk::CullModeFlagBits::eBack, vk::FrontFace::eClockwise, false, 0.0f, 0.0f, 0.0f, 1.0f};
        vk::PipelineMultisampleStateCreateInfo multisampling = {{}, vk::SampleCountFlagBits::e1, false, 1.0f, nullptr,
                false, false};
        vk::PipelineColorBlendAttachmentState colorBlendAttachment = {
                false,
                vk::BlendFactor::eOne,
                vk::BlendFactor::eZero,
                vk::BlendOp::eAdd,
                vk::BlendFactor::eOne,
                vk::BlendFactor::eZero,
                vk::BlendOp::eAdd, 
                vk::ColorComponentFlagBits::eR
                        | vk::ColorComponentFlagBits::eG
                        | vk::ColorComponentFlagBits::eB
                        | vk::ColorComponentFlagBits::eA
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending = {{}, false, vk::LogicOp::eCopy, 1U, &colorBlendAttachment,
                {0.0f, 0.0f, 0.0f, 0.0f}};
        vk::DynamicState dynamicStates[] = { vk::DynamicState::eViewport, vk::DynamicState::eLineWidth };
        vk::PipelineDynamicStateCreateInfo dynamicState = {{}, sizeof dynamicStates / sizeof(dynamicStates[0]),
                dynamicStates};

        mPipelineLayout = mDevice->createPipelineLayoutUnique({});

        mPipeline = mDevice->createGraphicsPipelineUnique(nullptr, {{}, 2U, shaderStages, &vertexInputInfo,
                &inputAssembly, nullptr, &viewportState, &rasterizer, &multisampling, nullptr, &colorBlending, nullptr,
                mPipelineLayout.get(), mRenderPass.get(), 0U, nullptr, -1});
        if (!mPipeline)
            throw std::runtime_error("failed to create pipeline!");
    }

    void createFramebuffers()
    {
        std::transform(mSwapchainImageViews.begin(), mSwapchainImageViews.end(),
                std::back_inserter(mSwapchainFramebuffers), [this] (auto& imageView) {
            vk::ImageView attachments[] = {
                imageView.get()
            };

            return mDevice->createFramebufferUnique({{}, mRenderPass.get(), 1U, attachments, mSwapchainExtent.width,
                    mSwapchainExtent.height, 1U});
        });
    }

    void createCommandPool()
    {
        auto indices = getQueueFamilyIndices(mPhysicalDevice);
        mCommandPool = mDevice->createCommandPoolUnique({{}, indices.graphicsFamily});
    }

    void createCommandBuffers()
    {
        mCommandBuffers = mDevice->allocateCommandBuffers({mCommandPool.get(), vk::CommandBufferLevel::ePrimary,
                (uint32_t) mSwapchainFramebuffers.size()});

        auto framebufferIt = mSwapchainFramebuffers.begin();
        for (auto& commandBuffer : mCommandBuffers)
        {
            commandBuffer.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse, nullptr});

            vk::ClearValue clearColor = (vk::ClearColorValue) {std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}};
            commandBuffer.beginRenderPass({mRenderPass.get(), (framebufferIt++)->get(), {{0, 0}, mSwapchainExtent}, 1U,
                    &clearColor}, vk::SubpassContents::eInline);

            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, mPipeline.get());
            commandBuffer.draw(3U, 1U, 0U, 0U);
            commandBuffer.endRenderPass();
            commandBuffer.end();
        }
    }

    void createSemaphores()
    {
        for (size_t i = 0; i < kMaxInFlight; i++)
        {
            if (!mImageAvailableSemaphores.emplace_back(mDevice->createSemaphoreUnique({}))
                    || !mRenderFinishedSemaphores.emplace_back(mDevice->createSemaphoreUnique({}))
                    || !mInFlightFences.emplace_back(mDevice->createFenceUnique({vk::FenceCreateFlagBits::eSignaled})))
                throw std::runtime_error("failed to create synchronization object for a frame!");
        }
    }

    void drawFrame()
    {
        mDevice->waitForFences(1, &mInFlightFences[currentFrame].get(), true, std::numeric_limits<uint64_t>::max());
        mDevice->resetFences(1, &mInFlightFences[currentFrame].get());

        auto res = mDevice->acquireNextImageKHR(mSwapchain.get(), std::numeric_limits<uint64_t>::max(),
                mImageAvailableSemaphores[currentFrame].get(), nullptr);
        if (res.result != vk::Result::eSuccess)
            throw std::runtime_error("failed to acquire image from swapchain!");

        uint32_t imageIndex = res.value;

        vk::Semaphore waitSemaphores[] = {mImageAvailableSemaphores[currentFrame].get()};
        vk::Semaphore signalSemaphores[] = {mRenderFinishedSemaphores[currentFrame].get()};
        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        vk::SubmitInfo submitInfo = {1U, waitSemaphores, waitStages, 1U, &mCommandBuffers[imageIndex], 1U,
                signalSemaphores};
        if (mGraphicsQueue.submit(1U, &submitInfo, mInFlightFences[currentFrame].get()) != vk::Result::eSuccess)
            throw std::runtime_error("failed to submit draw command buffer!");

        vk::SwapchainKHR swapchains[] = { mSwapchain.get() };
        mPresentQueue.presentKHR({1U, signalSemaphores, 1U, swapchains, &imageIndex, nullptr});

        currentFrame = (currentFrame + 1) % kMaxInFlight;
    }
};

int main()
{
    HelloTriangleApplication app;

    try
    {
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

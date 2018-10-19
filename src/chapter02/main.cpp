#define GLFW_INCLUDE_VULKAN
#define VULKAN_HPP_TYPESAFE_CONVERSION
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <thread>
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

struct Vertex
{
    glm::vec2 pos;
    glm::vec3 color;

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        return vk::VertexInputBindingDescription()
            .setBinding(0)
            .setStride(sizeof(Vertex))
            .setInputRate(vk::VertexInputRate::eVertex);
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescription()
    {
        return {
            vk::VertexInputAttributeDescription()
                    .setLocation(0)
                    .setBinding(0)
                    .setFormat(vk::Format::eR32G32Sfloat)
                    .setOffset(offsetof(Vertex, pos)),
            vk::VertexInputAttributeDescription()
                    .setLocation(1)
                    .setBinding(0)
                    .setFormat(vk::Format::eR32G32B32Sfloat)
                    .setOffset(offsetof(Vertex, color))
        };
    }
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

    const std::vector<Vertex> vertices = {
        {{ 0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
        {{ 0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}},
        {{-0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}}
    };

    GLFWwindow*                      mWindow;

    // application objects
    vk::UniqueInstance               mInstance;
    vk::UniqueDebugReportCallbackEXT mDebugReportCallback;
    vk::UniqueSurfaceKHR             mSurface;
    vk::PhysicalDevice               mPhysicalDevice;
    vk::UniqueDevice                 mDevice;
    vk::Queue                        mGraphicsQueue;
    vk::Queue                        mPresentQueue;
    vk::UniqueCommandPool            mCommandPool;
    vk::UniqueBuffer                 mVertexBuffer;
    vk::UniqueDeviceMemory           mVertexBufferMemory;
    vk::UniqueShaderModule           mVertexShaderModule;
    vk::UniqueShaderModule           mFragmentShaderModule;
    std::vector<vk::UniqueSemaphore> mImageAvailableSemaphores;
    std::vector<vk::UniqueSemaphore> mRenderFinishedSemaphores;
    std::vector<vk::UniqueFence>     mInFlightFences;

    // swapchain dependent objects
    vk::UniqueSwapchainKHR               mSwapchain;
    vk::Format                           mSwapchainImageFormat;
    vk::Extent2D                         mSwapchainExtent;
    std::vector<vk::Image>               mSwapchainImages;
    std::vector<vk::UniqueImageView>     mSwapchainImageViews;
    vk::UniqueRenderPass                 mRenderPass;
    vk::UniquePipelineLayout             mPipelineLayout;
    vk::UniquePipeline                   mPipeline;
    std::vector<vk::UniqueFramebuffer>   mSwapchainFramebuffers;
    std::vector<vk::UniqueCommandBuffer> mCommandBuffers; 

    size_t currentFrame = 0;
    bool framebufferResized = false;

public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        mWindow = glfwCreateWindow(kWidth, kHeight, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(mWindow, this);
        glfwSetFramebufferSizeCallback(mWindow, framebufferResizeCallback);
    }

    void initVulkan()
    {
        // application objects
        createInstance();
        createSurface();
        createDevice();
        createCommandPool();
        createVertexBuffers();
        createShaderModules();
        createSynchronizationObjects();

        // swapchain dependent objects
        createSwapchain();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandBuffers();
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

    void recreateSwapchain()
    {
        // handle minimization
        int width = 0, height = 0;
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(mWindow, &width, &height);
            glfwWaitEvents();
        }

        // wait for all device operations to complete before destroying stuff
        mDevice->waitIdle();

        // destroy swapchain
        mCommandBuffers.clear();
        mSwapchainFramebuffers.clear();
        mPipeline.reset();
        mPipelineLayout.reset();
        mRenderPass.reset();
        mSwapchainImageViews.clear();
        mSwapchain.reset();

        // create swapchain
        createSwapchain();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandBuffers();
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
            auto predicate = [layer] (auto& availableLayer)
            {
                return std::string_view(availableLayer.layerName) == std::string_view(layer);
            };

            if (std::find_if(availableLayers.begin(), availableLayers.end(), predicate) == availableLayers.end())
            {
                return false;
            }
        }
        return true;
    }

    std::vector<const char*> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (kValidationEnabled)
        {
            extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }

        return extensions;
    }

    QueueFamilyIndices getQueueFamilyIndices(vk::PhysicalDevice& d)
    {
        QueueFamilyIndices indices;

        auto queueFamilies = d.getQueueFamilyProperties();
        for (auto it = queueFamilies.begin(); it != queueFamilies.end(); it++)
        {
            uint32_t id = std::distance(queueFamilies.begin(), it);
            if (it->queueFlags & vk::QueueFlagBits::eGraphics)
            {
                indices.graphicsFamily = id;
            }

            vk::Bool32 supported;
            d.getSurfaceSupportKHR(id, mSurface.get(), &supported);
            if (supported)
            {
                indices.presentFamily = id;
            }
        }
        return indices;
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
    {
        auto memoryProperties = mPhysicalDevice.getMemoryProperties();

        for (uint32_t i{0}; i < memoryProperties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i)) && ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
                return i;
        }
    }

    vk::UniqueShaderModule createShaderModule(const std::string& path)
    {
        auto object = readFile(path);
        return mDevice->createShaderModuleUnique(
                vk::ShaderModuleCreateInfo()
                    .setCodeSize(object.size())
                    .setPCode(reinterpret_cast<const uint32_t*>(object.data())));
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
        {
            return { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };
        }

        for (const auto& availableFormat : formats)
        {
            if (availableFormat.format == vk::Format::eB8G8R8A8Unorm
                    && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            {
                return availableFormat;
            }
        }
        return formats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> presentModes)
    {
        vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;
        for (const auto& presentMode : presentModes)
        {
            if (presentMode == vk::PresentModeKHR::eMailbox)
            {
                return presentMode;
            }
            else if (presentMode == vk::PresentModeKHR::eImmediate)
            {
                bestMode = presentMode;
            }
        }
        return bestMode;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }

        int width, height;
        glfwGetFramebufferSize(mWindow, &width, &height);

        return {
            std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, (uint32_t) width)),
            std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height,
                    (uint32_t) height))
        };
    }

    void createInstance()
    {
        vk::ApplicationInfo appInfo("Hello Triangle", VK_MAKE_VERSION(0, 1, 0), "No Engine", VK_MAKE_VERSION(0, 1, 0),
                VK_API_VERSION_1_1);

        // check validation layer support
        if (kValidationEnabled && !checkValidationLayerSupport())
        {
            throw std::runtime_error("validation layers requrested, but not available!");
        }

        auto extensions = getRequiredExtensions();
        mInstance = vk::createInstanceUnique(
                vk::InstanceCreateInfo()
                    .setPApplicationInfo(&appInfo)
                    .setEnabledLayerCount(kValidationEnabled ? (uint32_t) validationLayers.size() : 0)
                    .setPpEnabledLayerNames(kValidationEnabled ? validationLayers.data() : nullptr)
                    .setEnabledExtensionCount((uint32_t) extensions.size())
                    .setPpEnabledExtensionNames(extensions.data()));
        if (!mInstance)
        {
            throw std::runtime_error("failed to create instance!");
        }

        if (kValidationEnabled)
        {
            setupDebugCallback();
        }
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
        mDebugReportCallback = mInstance->createDebugReportCallbackEXTUnique(
                vk::DebugReportCallbackCreateInfoEXT()
                    .setFlags(vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning)
                    .setPfnCallback(&debugCallback));
        if (!mDebugReportCallback)
        {
            throw std::runtime_error("failed to create debug report callback!");
        }
    }

    void createSurface()
    {
        vk::SurfaceKHR surface;
        vk::Result result = static_cast<vk::Result>(glfwCreateWindowSurface(mInstance.get(), mWindow, nullptr,
                reinterpret_cast<VkSurfaceKHR*>(&surface)));

        vk::ObjectDestroy<vk::Instance> deleter(mInstance.get(), nullptr);
        mSurface = vk::createResultValue(result, surface, "createSurface", deleter);
    }

    void createDevice()
    {
        auto devices = mInstance->enumeratePhysicalDevices();

        // Get the first device that supports required features
        for (auto& device : devices)
        {
            mPhysicalDevice = device;

            auto details = getSwapchainSupportDetails(device);
            if (details.formats.empty() || details.presentModes.empty())
            {
                continue;
            }

            // Assemble list of UNIQUE queue families to be created
            QueueFamilyIndices indices = getQueueFamilyIndices(device);
            const std::set<uint32_t> queueFamilies = { 
                indices.graphicsFamily,
                indices.presentFamily
            };

            // Transform queue families into queue create infos
            float defaultPriority = 1.0f;
            std::vector<vk::DeviceQueueCreateInfo> queues;
            std::transform(queueFamilies.begin(), queueFamilies.end(), std::back_inserter(queues),
                    [&defaultPriority] (auto family) -> vk::DeviceQueueCreateInfo
            {
                return vk::DeviceQueueCreateInfo()
                        .setQueueFamilyIndex(family)
                        .setQueueCount(1)
                        .setPQueuePriorities(&defaultPriority);
            });

            // note: device layers have been deprecated
            mDevice = device.createDeviceUnique(
                    vk::DeviceCreateInfo()
                        .setQueueCreateInfoCount((uint32_t) queues.size())
                        .setPQueueCreateInfos(queues.data())
                        .setEnabledExtensionCount((uint32_t) deviceExtensions.size())
                        .setPpEnabledExtensionNames(deviceExtensions.data()));
            if (!mDevice)
            {
                continue;
            }
            if (!(mGraphicsQueue = mDevice->getQueue(indices.graphicsFamily, 0)))
            {
                throw std::runtime_error("failed to get graphics queue!");
            }
            if (!(mPresentQueue = mDevice->getQueue(indices.presentFamily, 0)))
            {
                throw std::runtime_error("failed to get present queue!");
            }
            return;
        }
        throw std::runtime_error("failed to create logical device!");
    }

    void createCommandPool()
    {
        auto indices = getQueueFamilyIndices(mPhysicalDevice);
        mCommandPool = mDevice->createCommandPoolUnique(
                vk::CommandPoolCreateInfo()
                    .setQueueFamilyIndex(indices.graphicsFamily));
        if (!mCommandPool)
        {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void createVertexBuffers()
    {
        mVertexBuffer = mDevice->createBufferUnique(
                vk::BufferCreateInfo()
                    .setSize(sizeof vertices[0] * vertices.size())
                    .setUsage(vk::BufferUsageFlagBits::eVertexBuffer)
                    .setSharingMode(vk::SharingMode::eExclusive));
        if (!mVertexBuffer)
        {
            throw std::runtime_error("failed to create vertex buffer!");
        }

        auto memoryRequirements = mDevice->getBufferMemoryRequirements(mVertexBuffer.get());
        mVertexBufferMemory = mDevice->allocateMemoryUnique(
                vk::MemoryAllocateInfo()
                    .setAllocationSize(memoryRequirements.size)
                    .setMemoryTypeIndex(
                        findMemoryType(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible
                            | vk::MemoryPropertyFlagBits::eHostCoherent)));
        if (!mVertexBufferMemory)
        {
            throw std::runtime_error("failed to create buffer memory!");
        }

        mDevice->bindBufferMemory(mVertexBuffer.get(), mVertexBufferMemory.get(), 0);
        Vertex* vertexBufferData = static_cast<Vertex*>(mDevice->mapMemory(mVertexBufferMemory.get(), 0,
                sizeof vertices[0] * vertices.size(), {}));
        if (!vertexBufferData)
        {
            throw std::runtime_error("failed to map buffer memory!");
        }

        std::copy(vertices.begin(), vertices.end(), vertexBufferData);        
        mDevice->unmapMemory(mVertexBufferMemory.get());     
    }

    void createShaderModules()
    {
        if (!((mVertexShaderModule = createShaderModule("shaders/chapter02/triangle.vert.spv")))
                || !((mFragmentShaderModule = createShaderModule("shaders/chapter02/triangle.frag.spv"))))
        {
            throw std::runtime_error("failed to shader create modules!");
        }
    }

    void createSynchronizationObjects()
    {
        for (size_t i = 0; i < kMaxInFlight; i++)
        {
            if (!mImageAvailableSemaphores.emplace_back(mDevice->createSemaphoreUnique({}))
                    || !mRenderFinishedSemaphores.emplace_back(mDevice->createSemaphoreUnique({}))
                    || !mInFlightFences.emplace_back(mDevice->createFenceUnique({vk::FenceCreateFlagBits::eSignaled})))
                throw std::runtime_error("failed to create synchronization object for a frame!");
        }
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

        auto indices = getQueueFamilyIndices(mPhysicalDevice);
        const std::vector<uint32_t> queueFamilyIndices = { indices.graphicsFamily, indices.presentFamily };
        mSwapchain = mDevice->createSwapchainKHRUnique(
                vk::SwapchainCreateInfoKHR()
                    .setSurface(mSurface.get())
                    .setMinImageCount(imageCount)
                    .setImageFormat(surfaceFormat.format)
                    .setImageColorSpace(surfaceFormat.colorSpace)
                    .setImageExtent(extent)
                    .setImageArrayLayers(1)
                    .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
                    .setImageSharingMode((indices.graphicsFamily != indices.presentFamily)
                        ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive)
                    .setQueueFamilyIndexCount((indices.graphicsFamily != indices.presentFamily)
                        ? (uint32_t) queueFamilyIndices.size() : 0)
                    .setPQueueFamilyIndices((indices.graphicsFamily != indices.presentFamily)
                        ? queueFamilyIndices.data() : nullptr)
                    .setPreTransform(swapchainSupportDetails.capabilities.currentTransform)
                    .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
                    .setPresentMode(presentMode)
                    .setClipped(true)
                    .setOldSwapchain(nullptr));
        if (!mSwapchain)
        {
            throw std::runtime_error("failed to create swapchain!");
        }

        mSwapchainImageFormat = surfaceFormat.format;
        mSwapchainExtent = extent;

        // create image views from all swapchain imagesc
        mSwapchainImages = mDevice->getSwapchainImagesKHR(mSwapchain.get());
        std::transform(mSwapchainImages.begin(), mSwapchainImages.end(), std::back_inserter(mSwapchainImageViews),
                [this] (auto& image)
        {
            return mDevice->createImageViewUnique(
                    vk::ImageViewCreateInfo()
                        .setImage(image)
                        .setViewType(vk::ImageViewType::e2D)
                        .setFormat(mSwapchainImageFormat)
                        .setSubresourceRange(
                            vk::ImageSubresourceRange()
                                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                .setBaseMipLevel(0)
                                .setLevelCount(1)
                                .setBaseArrayLayer(0)
                                .setLayerCount(1)));
        });
    }

    void createRenderPass()
    {
        auto colorAttachment = vk::AttachmentDescription()
                .setFormat(mSwapchainImageFormat)
                .setSamples(vk::SampleCountFlagBits::e1)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                .setInitialLayout(vk::ImageLayout::eUndefined)
                .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

        auto colorAttachmentRef = vk::AttachmentReference()
                .setAttachment(0)
                .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

        auto subpass = vk::SubpassDescription()
                .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                .setColorAttachmentCount(1)
                .setPColorAttachments(&colorAttachmentRef);

        auto subpassDependency = vk::SubpassDependency()
                .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                .setSrcAccessMask({})
                .setDstSubpass(0)
                .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

        mRenderPass = mDevice->createRenderPassUnique(
                vk::RenderPassCreateInfo()
                    .setAttachmentCount(1)
                    .setPAttachments(&colorAttachment)
                    .setSubpassCount(1)
                    .setPSubpasses(&subpass)
                    .setDependencyCount(1)
                    .setPDependencies(&subpassDependency));
        if (!mRenderPass)
        {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createGraphicsPipeline()
    {
        vk::PipelineShaderStageCreateInfo shaderStages[] = {
            vk::PipelineShaderStageCreateInfo()
                    .setStage(vk::ShaderStageFlagBits::eVertex)
                    .setModule(mVertexShaderModule.get())
                    .setPName("main"),
            vk::PipelineShaderStageCreateInfo()
                    .setStage(vk::ShaderStageFlagBits::eFragment)
                    .setModule(mFragmentShaderModule.get())
                    .setPName("main")
        };

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescription = Vertex::getAttributeDescription();
        auto vertexInputInfo = vk::PipelineVertexInputStateCreateInfo()
                .setVertexBindingDescriptionCount(1)
                .setPVertexBindingDescriptions(&bindingDescription)
                .setVertexAttributeDescriptionCount(attributeDescription.size())
                .setPVertexAttributeDescriptions(attributeDescription.data());

        auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo()
                .setTopology(vk::PrimitiveTopology::eTriangleList)
                .setPrimitiveRestartEnable(false);

        auto viewport = vk::Viewport()
                .setX(0.f)
                .setY(0.f)
                .setWidth(mSwapchainExtent.width)
                .setHeight(mSwapchainExtent.height)
                .setMinDepth(0.f)
                .setMaxDepth(0.f);

        auto scissor = vk::Rect2D()
                .setExtent(mSwapchainExtent);

        auto viewportState = vk::PipelineViewportStateCreateInfo()
                .setViewportCount(1)
                .setPViewports(&viewport)
                .setScissorCount(1)
                .setPScissors(&scissor);

        auto rasterizer = vk::PipelineRasterizationStateCreateInfo()
                .setDepthClampEnable(false)
                .setDepthBiasEnable(false)
                .setDepthBiasClamp(0.f)
                .setDepthBiasConstantFactor(0.f)
                .setDepthBiasSlopeFactor(0.f)
                .setRasterizerDiscardEnable(false)
                .setPolygonMode(vk::PolygonMode::eFill)
                .setCullMode(vk::CullModeFlagBits::eBack)
                .setFrontFace(vk::FrontFace::eClockwise)
                .setLineWidth(1.0f);

        auto multisampling = vk::PipelineMultisampleStateCreateInfo()
                .setRasterizationSamples(vk::SampleCountFlagBits::e1)
                .setSampleShadingEnable(false)
                .setMinSampleShading(1.f)
                .setPSampleMask(nullptr)
                .setAlphaToCoverageEnable(false)
                .setAlphaToOneEnable(false);

        auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState()
                .setBlendEnable(false)
                .setSrcColorBlendFactor(vk::BlendFactor::eOne)
                .setDstColorBlendFactor(vk::BlendFactor::eZero)
                .setColorBlendOp(vk::BlendOp::eAdd)
                .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
                .setDstAlphaBlendFactor(vk::BlendFactor::eZero)
                .setAlphaBlendOp(vk::BlendOp::eAdd)
                .setColorWriteMask(vk::ColorComponentFlagBits::eR 
                    | vk::ColorComponentFlagBits::eG 
                    | vk::ColorComponentFlagBits::eB 
                    | vk::ColorComponentFlagBits::eA);

        auto colorBlending = vk::PipelineColorBlendStateCreateInfo()
                .setLogicOpEnable(false)
                .setLogicOp(vk::LogicOp::eCopy)
                .setAttachmentCount(1)
                .setPAttachments(&colorBlendAttachment)
                .setBlendConstants({0.0f, 0.0f, 0.0f, 0.0f});

        vk::DynamicState dynamicStates[] = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eLineWidth
        };

        auto dynamicState = vk::PipelineDynamicStateCreateInfo()
                .setDynamicStateCount(sizeof dynamicStates / sizeof dynamicStates[0])
                .setPDynamicStates(dynamicStates);

        mPipelineLayout = mDevice->createPipelineLayoutUnique({});
        mPipeline = mDevice->createGraphicsPipelineUnique(
                nullptr,
                vk::GraphicsPipelineCreateInfo()
                    .setStageCount(2)
                    .setPStages(shaderStages)
                    .setPVertexInputState(&vertexInputInfo)
                    .setPInputAssemblyState(&inputAssembly)
                    .setPViewportState(&viewportState)
                    .setPRasterizationState(&rasterizer)
                    .setPMultisampleState(&multisampling)
                    .setPColorBlendState(&colorBlending)
                    .setLayout(mPipelineLayout.get())
                    .setRenderPass(mRenderPass.get())
                    .setBasePipelineIndex(-1));
        if (!mPipeline)
        {
            throw std::runtime_error("failed to create pipeline!");
        }
    }

    void createFramebuffers()
    {
        std::transform(mSwapchainImageViews.begin(), mSwapchainImageViews.end(),
                std::back_inserter(mSwapchainFramebuffers), [this] (auto& imageView)
        {
            vk::ImageView attachments[] = {
                imageView.get()
            };

            return mDevice->createFramebufferUnique(
                    vk::FramebufferCreateInfo()
                        .setRenderPass(mRenderPass.get())
                        .setAttachmentCount(1)
                        .setPAttachments(attachments)
                        .setWidth(mSwapchainExtent.width)
                        .setHeight(mSwapchainExtent.height)
                        .setLayers(1));
        });
    }

    void createCommandBuffers()
    {
        mCommandBuffers = mDevice->allocateCommandBuffersUnique(
                vk::CommandBufferAllocateInfo()
                    .setCommandPool(mCommandPool.get())
                    .setLevel(vk::CommandBufferLevel::ePrimary)
                    .setCommandBufferCount(mSwapchainFramebuffers.size())); 

        auto framebufferIt = mSwapchainFramebuffers.begin();
        for (auto& commandBuffer : mCommandBuffers)
        {
            commandBuffer->begin(
                    vk::CommandBufferBeginInfo()
                        .setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse));

            vk::ClearValue clearColor = vk::ClearColorValue {std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}};
            commandBuffer->beginRenderPass(
                    vk::RenderPassBeginInfo()
                        .setRenderPass(mRenderPass.get())
                        .setFramebuffer((framebufferIt++)->get())
                        .setRenderArea({{0, 0}, mSwapchainExtent})
                        .setPClearValues(&clearColor)
                        .setClearValueCount(1U),
                    vk::SubpassContents::eInline);

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, mPipeline.get());

            vk::Buffer vertexBuffers[] = {mVertexBuffer.get()};
            vk::DeviceSize offsets[] = {0};
            commandBuffer->bindVertexBuffers(0U, 1U, vertexBuffers, offsets);

            commandBuffer->draw(3U, 1U, 0U, 0U);
            commandBuffer->endRenderPass();
            commandBuffer->end();
        }
    }

    void drawFrame()
    {
        mDevice->waitForFences(1, &mInFlightFences[currentFrame].get(), true, std::numeric_limits<uint64_t>::max());

        // attempt to acquire an image to write into
        uint32_t imageIndex = std::numeric_limits<uint32_t>::max();
        try
        {
            auto res = mDevice->acquireNextImageKHR(mSwapchain.get(), std::numeric_limits<uint64_t>::max(),
                    mImageAvailableSemaphores[currentFrame].get(), nullptr);
            if (res.result != vk::Result::eSuccess)
                throw std::runtime_error("failed to acquire image from swapchain!");

            imageIndex = res.value;
        }
        catch (vk::OutOfDateKHRError& error)
        {
            // reset fence since we acquired content into it
            recreateSwapchain();
            return;
        }

        // submit commandbuffer
        vk::Semaphore waitSemaphores[] = {mImageAvailableSemaphores[currentFrame].get()};
        vk::Semaphore signalSemaphores[] = {mRenderFinishedSemaphores[currentFrame].get()};
        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        vk::SubmitInfo submitInfo = {1U, waitSemaphores, waitStages, 1U, &mCommandBuffers[imageIndex].get(), 1U,
                signalSemaphores};

        mDevice->resetFences(1, &mInFlightFences[currentFrame].get());
        if (mGraphicsQueue.submit(1U, &submitInfo, mInFlightFences[currentFrame].get()) != vk::Result::eSuccess)
            throw std::runtime_error("failed to submit draw command buffer!");

        // present image
        try
        {
            vk::SwapchainKHR swapchains[] = { mSwapchain.get() };
            auto res = mPresentQueue.presentKHR({1U, signalSemaphores, 1U, swapchains, &imageIndex, nullptr});
            if (res == vk::Result::eSuboptimalKHR || framebufferResized)
                throw vk::OutOfDateKHRError("suboptimal");
            else if (res != vk::Result::eSuccess)
                throw std::runtime_error("failed to present swapchain image!");
        }
        catch (vk::OutOfDateKHRError& error)
        {
            framebufferResized = false;
            recreateSwapchain();
        }

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

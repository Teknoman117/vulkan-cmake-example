#define GLFW_INCLUDE_VULKAN
#define VULKAN_HPP_TYPESAFE_CONVERSION
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <iostream>
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
}

class HelloTriangleApplication
{
    constexpr static int kWidth = 1920;
    constexpr static int kHeight = 1080;

#ifdef NDEBUG
    constexpr static bool kValidationEnabled = false;
#else
    constexpr static bool kValidationEnabled = true;
#endif

    struct QueueFamilyIndices {
        int graphicsFamily = -1;
        int presentFamily = -1;

        constexpr bool isComplete() {
            return graphicsFamily >= 0 && presentFamily >= 0;
        }
    };

    // collect list of desired layers
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_LUNARG_standard_validation"
    };

    GLFWwindow* mWindow;
    vk::UniqueInstance mInstance;
    vk::UniqueDebugReportCallbackEXT mDebugReportCallback;
    vk::UniqueDevice mDevice;
    vk::Queue mGraphicsQueue;
    vk::Queue mPresentQueue;

    vk::UniqueSurfaceKHR mSurface;

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
        // Find a suitable GPU
        auto devices = mInstance->enumeratePhysicalDevices();
        auto device = std::find_if(devices.begin(), devices.end(), [this] (vk::PhysicalDevice& d)
        {
            QueueFamilyIndices indices = getQueueFamilyIndices(d);
            auto properties = d.getProperties();
            auto features = d.getFeatures();

            return properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu
                    && features.geometryShader
                    && indices.isComplete();
        });

        if (device == devices.end())
            throw std::runtime_error("failed to find a suitable GPU!");

        std::cerr << "[INFO] Using Device: " << device->getProperties().deviceName << std::endl;

        // Assemble list of UNIQUE queue families to be created
        QueueFamilyIndices indices = getQueueFamilyIndices(*device);
        float defaultPriority = 1.0f;
        const std::set<int> queueFamilies = {
            indices.graphicsFamily,
            indices.presentFamily
        };

        std::vector<vk::DeviceQueueCreateInfo> queues;
        for (int family : queueFamilies)
            queues.push_back(vk::DeviceQueueCreateInfo({}, indices.graphicsFamily, 1, &defaultPriority));

        // note: device layers have been deprecated
        mDevice = device->createDeviceUnique({{}, (uint32_t) queues.size(), queues.data()});
        if (!mDevice)
            throw std::runtime_error("failed to create logical device!");

        mGraphicsQueue = mDevice->getQueue(indices.graphicsFamily, 0);
        if (!mGraphicsQueue)
            throw std::runtime_error("failed to get graphics queue!");

        mPresentQueue = mDevice->getQueue(indices.presentFamily, 0);
        if (!mPresentQueue)
            throw std::runtime_error("failed to get present queue!");
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(mWindow))
        {
            glfwPollEvents();
        }
    }

    void cleanup()
    {
        glfwDestroyWindow(mWindow);
        glfwTerminate();
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

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <Cocoa/Cocoa.h>

namespace {

id g_magnifyMonitor = nil;
NSWindow* g_targetWindow = nil;
double g_pendingMagnify = 0.0;

}  // namespace

extern "C" void MandelbrotInstallMacGestureBridge(GLFWwindow* window) {
  g_targetWindow = glfwGetCocoaWindow(window);
  if (g_magnifyMonitor != nil) {
    return;
  }

  g_magnifyMonitor = [NSEvent addLocalMonitorForEventsMatchingMask:NSEventMaskMagnify
                                                           handler:^NSEvent* (NSEvent* event) {
                                                             if (g_targetWindow == nil || [event window] == g_targetWindow) {
                                                               g_pendingMagnify += [event magnification];
                                                               return nil;
                                                             }
                                                             return event;
                                                           }];
}

extern "C" double MandelbrotConsumeMacMagnifyDelta() {
  const double delta = g_pendingMagnify;
  g_pendingMagnify = 0.0;
  return delta;
}

extern "C" void MandelbrotShutdownMacGestureBridge() {
  if (g_magnifyMonitor != nil) {
    [NSEvent removeMonitor:g_magnifyMonitor];
    g_magnifyMonitor = nil;
  }
  g_targetWindow = nil;
  g_pendingMagnify = 0.0;
}

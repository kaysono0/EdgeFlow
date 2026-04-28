#import <Foundation/Foundation.h>

typedef NS_ENUM(NSInteger, NativeInferenceFramework) {
    NativeInferenceFrameworkMNN NS_SWIFT_NAME(mnn) = 0,
    NativeInferenceFrameworkNCNN NS_SWIFT_NAME(ncnn) = 1,
} NS_SWIFT_NAME(NativeInferenceFramework);

typedef NS_ENUM(NSInteger, NativeInferenceBackend) {
    NativeInferenceBackendCPU NS_SWIFT_NAME(cpu) = 0,
    NativeInferenceBackendMetal NS_SWIFT_NAME(metal) = 1,
    NativeInferenceBackendVulkan NS_SWIFT_NAME(vulkan) = 2,
} NS_SWIFT_NAME(NativeInferenceBackend);

NS_ASSUME_NONNULL_BEGIN

@interface NativeInferenceBridge : NSObject

- (NSData * _Nullable)runBenchmarkForFramework:(NativeInferenceFramework)framework
                                       backend:(NativeInferenceBackend)backend
                                     imagePath:(NSString * _Nullable)imagePath
                                         error:(NSError * _Nullable * _Nullable)error;

@end

NS_ASSUME_NONNULL_END

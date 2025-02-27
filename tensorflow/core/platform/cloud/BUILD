# Description:
# Cloud file system implementation.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/tsl:tsl.bzl",
    "tsl_copts",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":dependency_whitelist",
    ],
    licenses = ["notice"],
)

package_group(
    name = "dependency_whitelist",
    packages = [
        "//learning/brain/tfrc/...",
        "//tensorflow/...",
        "//third_party/gsmemcachedfs/...",
        "//third_party/py/tensorflow_io/...",
    ],
)

cc_library(
    name = "expiring_lru_cache",
    hdrs = ["expiring_lru_cache.h"],
    copts = tsl_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/tsl/platform/cloud:expiring_lru_cache",
    ],
)

cc_library(
    name = "file_block_cache",
    hdrs = ["file_block_cache.h"],
    copts = tsl_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/tsl/platform/cloud:file_block_cache",
    ],
)

cc_library(
    name = "ram_file_block_cache",
    hdrs = ["ram_file_block_cache.h"],
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":file_block_cache",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/tsl/platform/cloud:ram_file_block_cache",
    ],
)

cc_library(
    name = "gcs_dns_cache",
    hdrs = ["gcs_dns_cache.h"],
    copts = tsl_copts(),
    deps = [
        ":http_request",
        "//tensorflow/core:lib",
        "//tensorflow/tsl/platform/cloud:gcs_dns_cache",
    ],
)

cc_library(
    name = "gcs_throttle",
    hdrs = ["gcs_throttle.h"],
    copts = tsl_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/tsl/platform/cloud:gcs_throttle",
    ],
)

cc_library(
    name = "gcs_file_system",
    hdrs = ["gcs_file_system.h"],
    copts = tsl_copts(),
    linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
    visibility = ["//visibility:public"],
    deps = [
        ":compute_engine_metadata_client",
        ":compute_engine_zone_provider",
        ":curl_http_request",
        ":expiring_lru_cache",
        ":file_block_cache",
        ":gcs_dns_cache",
        ":gcs_throttle",
        ":google_auth_provider",
        ":http_request",
        ":ram_file_block_cache",
        ":time_util",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:numbers",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:retrying_file_system",
        "//tensorflow/core/platform:retrying_utils",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:stringprintf",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/tsl/platform/cloud:gcs_file_system",
        "@jsoncpp_git//:jsoncpp",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gcs_file_system_disabled",
    hdrs = ["gcs_file_system.h"],
    copts = tsl_copts() + [
        "-DTPU_GCS_FS",
        "-DPLATFORM_CLOUD_TPU",
    ],
    linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
    visibility = ["//visibility:public"],
    deps = [
        ":compute_engine_metadata_client",
        ":compute_engine_zone_provider",
        ":curl_http_request",
        ":expiring_lru_cache",
        ":file_block_cache",
        ":gcs_dns_cache",
        ":gcs_throttle",
        ":google_auth_provider",
        ":http_request",
        ":ram_file_block_cache",
        ":time_util",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:numbers",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:retrying_file_system",
        "//tensorflow/core/platform:retrying_utils",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:stringprintf",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/tsl/platform/cloud:gcs_file_system_disabled",
        "@jsoncpp_git//:jsoncpp",
    ],
    alwayslink = 1,
)

cc_library(
    name = "http_request",
    hdrs = ["http_request.h"],
    copts = tsl_copts(),
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/tsl/platform/cloud:http_request",
    ],
)

cc_library(
    name = "curl_http_request",
    hdrs = ["curl_http_request.h"],
    copts = tsl_copts(),
    deps = [
        ":http_request",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:scanner",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/util:env_var",
        "//tensorflow/tsl/platform/cloud:curl_http_request",
        "@curl",
    ],
)

cc_library(
    name = "http_request_fake",
    testonly = 1,
    hdrs = [
        "http_request_fake.h",
    ],
    copts = tsl_copts(),
    deps = [
        ":curl_http_request",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/tsl/platform/cloud:http_request_fake",
        "@curl",
    ],
)

cc_library(
    name = "google_auth_provider",
    hdrs = [
        "auth_provider.h",
        "google_auth_provider.h",
    ],
    copts = tsl_copts(),
    deps = [
        ":compute_engine_metadata_client",
        ":oauth_client",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:base64",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:retrying_utils",
        "//tensorflow/core/platform:status",
        "//tensorflow/tsl/platform/cloud:google_auth_provider",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
    ],
)

cc_library(
    name = "compute_engine_metadata_client",
    hdrs = [
        "compute_engine_metadata_client.h",
    ],
    copts = tsl_copts(),
    deps = [
        ":curl_http_request",
        ":http_request",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:retrying_utils",
        "//tensorflow/tsl/platform/cloud:compute_engine_metadata_client",
    ],
)

cc_library(
    name = "compute_engine_zone_provider",
    hdrs = [
        "compute_engine_zone_provider.h",
        "zone_provider.h",
    ],
    copts = tsl_copts(),
    deps = [
        ":compute_engine_metadata_client",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/tsl/platform/cloud:compute_engine_zone_provider",
    ],
)

cc_library(
    name = "now_seconds_env",
    testonly = 1,
    hdrs = ["now_seconds_env.h"],
    copts = tsl_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/tsl/platform/cloud:now_seconds_env",
    ],
)

cc_library(
    name = "oauth_client",
    hdrs = [
        "oauth_client.h",
    ],
    copts = tsl_copts(),
    deps = [
        ":curl_http_request",
        ":http_request",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:base64",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/tsl/platform/cloud:oauth_client",
        "@boringssl//:crypto",
        "@jsoncpp_git//:jsoncpp",
    ],
)

cc_library(
    name = "time_util",
    hdrs = [
        "time_util.h",
    ],
    copts = tsl_copts(),
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/tsl/platform/cloud:time_util",
    ],
)

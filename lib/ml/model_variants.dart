enum ModelVariant { fp32, dynamic, int8 }

extension ModelVariantX on ModelVariant {
  String get title {
    switch (this) {
      case ModelVariant.fp32:
        return "FP32 (Most accurate)";
      case ModelVariant.dynamic:
        return "Dynamic (Recommended)";
      case ModelVariant.int8:
        return "INT8 (Fast, low accuracy)";
    }
  }

  String get assetPath {
    switch (this) {
      case ModelVariant.fp32:
        return "assets/models/tomato_fp32.tflite";
      case ModelVariant.dynamic:
        return "assets/models/tomato_dynamic.tflite";
      case ModelVariant.int8:
        return "assets/models/tomato_int8.tflite";
    }
  }
}

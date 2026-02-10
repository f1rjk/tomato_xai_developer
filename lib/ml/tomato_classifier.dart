import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'confidence.dart';
import 'image_preprocess.dart';
import 'model_variants.dart';

class Prediction {
  final String label;
  final double confidence;
  final int index;
  final List<double> probs;
  final bool abstained;
  final String? note;

  Prediction({
    required this.label,
    required this.confidence,
    required this.index,
    required this.probs,
    this.abstained = false,
    this.note,
  });
}

class TomatoClassifier {
  static const int imgSize = 224;

  final ConfidencePolicy policy;
  TomatoClassifier({this.policy = const ConfidencePolicy()});

  Interpreter? _interpreter;
  late List<String> _labels;
  ModelVariant _variant = ModelVariant.dynamic;

  Interpreter get interpreter => _interpreter!;
  List<String> get labels => _labels;
  ModelVariant get variant => _variant;

  Future<void> load({required ModelVariant variant}) async {
    _variant = variant;

    // Close old interpreter if exists
    _interpreter?.close();

    // Load new interpreter
    _interpreter = await Interpreter.fromAsset(variant.assetPath);

    // Load labels
    final labelsJson =
    await rootBundle.loadString('assets/models/class_names.json');
    final decoded = json.decode(labelsJson) as List<dynamic>;
    _labels = decoded.map((e) => e.toString()).toList();
  }

  Prediction predict(Uint8List imageBytes) {
    final resized = ImagePreprocess.decodeCropResize(imageBytes);

    final probs = _runProbsFromResized(resized);

    // Argmax
    int best = 0;
    double bestScore = probs[0];
    for (int i = 1; i < probs.length; i++) {
      if (probs[i] > bestScore) {
        bestScore = probs[i];
        best = i;
      }
    }

    final abstain = policy.shouldAbstain(probs);
    if (abstain) {
      return Prediction(
        label: "Uncertain",
        confidence: bestScore,
        index: best,
        probs: probs,
        abstained: true,
        note: "Not confident. Try a clearer close-up leaf photo with good light.",
      );
    }

    return Prediction(
      label: _labels[best],
      confidence: bestScore,
      index: best,
      probs: probs,
    );
  }

  /// Used by XAI too (occlusion repeatedly calls this)
  List<double> runProbsForImage(img.Image resized224) => _runProbsFromResized(resized224);

  List<double> _runProbsFromResized(img.Image resized224) {
    final inTensor = interpreter.getInputTensor(0);
    final outTensor = interpreter.getOutputTensor(0);

    final TensorType inType = inTensor.type;   // ✅ TensorType (not TfLiteType)
    final TensorType outType = outTensor.type;

    final inParams = inTensor.params;   // has scale & zeroPoint for quant models
    final outParams = outTensor.params;

    final input = _makeInput4D(
      resized224,
      inType,
      inParams.scale,
      inParams.zeroPoint,
    );

    final output = _makeOutput2D(labels.length, outType);

    interpreter.run(input, output);

    return _decodeOutput(output, outType, outParams.scale, outParams.zeroPoint);
  }

  Object _makeInput4D(
      img.Image image,
      TensorType type,
      double scale,
      int zeroPoint,
      ) {
    // If quant params are missing/0, fallback safely
    final double s = (scale == 0.0) ? (1.0 / 255.0) : scale;

    num q(double v01) {
      if (type == TensorType.float32) return v01;

      // Quantize float (0..1) -> int8/uint8 using scale/zeroPoint
      final raw = (v01 / s + zeroPoint).round();

      if (type == TensorType.uint8) return raw.clamp(0, 255);
      if (type == TensorType.int8) return raw.clamp(-128, 127);

      // If some unexpected type, still provide float
      return v01;
    }

    // Shape: [1][224][224][3]
    return [
      List.generate(imgSize, (y) {
        return List.generate(imgSize, (x) {
          final p = image.getPixel(x, y);

          final r01 = p.r / 255.0;
          final g01 = p.g / 255.0;
          final b01 = p.b / 255.0;

          return [q(r01), q(g01), q(b01)];
        });
      })
    ];
  }

  Object _makeOutput2D(int n, TensorType outType) {
    // Shape: [1][numClasses]
    if (outType == TensorType.float32) {
      return [List<double>.filled(n, 0.0)];
    }
    if (outType == TensorType.uint8 || outType == TensorType.int8) {
      return [List<int>.filled(n, 0)];
    }

    // Fallback
    return [List<double>.filled(n, 0.0)];
  }

  List<double> _decodeOutput(
      Object out2d,
      TensorType outType,
      double scale,
      int zeroPoint,
      ) {
    if (outType == TensorType.float32) {
      return ((out2d as List)[0] as List<double>);
    }

    // If quant params missing, fallback safely
    final double s = (scale == 0.0) ? (1.0 / 255.0) : scale;

    if (outType == TensorType.uint8 || outType == TensorType.int8) {
      final row = ((out2d as List)[0] as List<int>);
      return row.map((q) => (q - zeroPoint) * s).toList();
    }

    // Fallback
    return ((out2d as List)[0] as List<double>);
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }
}

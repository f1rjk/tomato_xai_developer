import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'image_preprocess.dart';

class GateResult {
  final bool isTomatoLeaf;
  final double tomatoScore; // tomato leaf confidence (0..1)
  final List<double> probs; // raw output vector
  final String note;

  const GateResult({
    required this.isTomatoLeaf,
    required this.tomatoScore,
    required this.probs,
    required this.note,
  });
}

class GateClassifier {
  static const int imgSize = 224;
  static const String assetPath = "assets/models/gate_dynamic.tflite";

  Interpreter? _interpreter;

  Future<void> load() async {
    _interpreter?.close();
    _interpreter = await Interpreter.fromAsset(assetPath);
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }

  Interpreter get interpreter {
    final it = _interpreter;
    if (it == null) throw Exception("Gate model not loaded. Call await gate.load()");
    return it;
  }

  /// Supports output shapes:
  /// - [1,1] sigmoid -> tomatoScore = probs[0]
  /// - [1,2] softmax -> tomatoScore = probs[tomatoIndex] (default 1)
// In gate_classifier.dart, modify the predict method:
  GateResult predict(
      Uint8List imageBytes, {
        double tau = 0.70,
        int tomatoIndex = 1,
      }) {
    final resized = ImagePreprocess.decodeCropResize(imageBytes);

    final inTensor = interpreter.getInputTensor(0);
    final outTensor = interpreter.getOutputTensor(0);

    final inType = inTensor.type;
    final outType = outTensor.type;

    final inParams = inTensor.params;
    final outParams = outTensor.params;

    final input = _makeInput4D(resized, inType, inParams.scale, inParams.zeroPoint);

    final outShape = outTensor.shape;
    final outN = (outShape.isNotEmpty) ? outShape.last : 1;

    final output = _makeOutput2D(outN, outType);

    interpreter.run(input, output);

    final probs = _decodeOutput(output, outType, outParams.scale, outParams.zeroPoint);

    // 🔍 DEBUG: Print raw probabilities
    print("🔍 Gate raw probs: $probs");
    print("🔍 Gate output shape: $outShape");
    print("🔍 Gate input type: $inType, output type: $outType");

    double tomatoScore;
    if (probs.length == 1) {
      tomatoScore = probs[0];
    } else {
      final idx = tomatoIndex.clamp(0, probs.length - 1);
      tomatoScore = probs[idx];
    }

    print("🔍 Gate tomato score: $tomatoScore (threshold: $tau)");

    final isTomato = tomatoScore >= tau;

    return GateResult(
      isTomatoLeaf: isTomato,
      tomatoScore: tomatoScore,
      probs: probs,
      note: isTomato
          ? "Tomato leaf detected"
          : "Not a tomato leaf / out of scope. Use a close-up tomato leaf photo.",
    );
  }
  // ---------------- helpers ----------------

  Object _makeInput4D(img.Image image, TensorType type, double scale, int zeroPoint) {
    final double s = (scale == 0.0) ? (1.0 / 255.0) : scale;

    num q(double v01) {
      if (type == TensorType.float32) return v01;

      final raw = (v01 / s + zeroPoint).round();
      if (type == TensorType.uint8) return raw.clamp(0, 255);
      if (type == TensorType.int8) return raw.clamp(-128, 127);

      return v01;
    }

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
    if (outType == TensorType.float32) return [List<double>.filled(n, 0.0)];
    if (outType == TensorType.uint8 || outType == TensorType.int8) return [List<int>.filled(n, 0)];
    return [List<double>.filled(n, 0.0)];
  }

  List<double> _decodeOutput(Object out2d, TensorType outType, double scale, int zeroPoint) {
    if (outType == TensorType.float32) {
      return ((out2d as List)[0] as List<double>);
    }

    final double s = (scale == 0.0) ? (1.0 / 255.0) : scale;

    if (outType == TensorType.uint8 || outType == TensorType.int8) {
      final row = ((out2d as List)[0] as List<int>);
      return row.map((q) => (q - zeroPoint) * s).toList();
    }

    return ((out2d as List)[0] as List<double>);
  }
}

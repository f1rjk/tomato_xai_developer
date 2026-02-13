import 'dart:convert';
import 'dart:math';
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
  double _temperature = 1.0; // Default, will be loaded from temperature.json

  Interpreter get interpreter => _interpreter!;
  List<String> get labels => _labels;
  ModelVariant get variant => _variant;
  double get temperature => _temperature;

  Future<void> load({required ModelVariant variant}) async {
    _variant = variant;

    _interpreter?.close();
    _interpreter = null;

    _interpreter = await Interpreter.fromAsset(variant.assetPath);

    // Load class names
    final labelsJson = await rootBundle.loadString('assets/models/class_names.json');
    final decoded = json.decode(labelsJson) as List<dynamic>;
    _labels = decoded.map((e) => e.toString()).toList();

    // Load temperature scaling factor
    try {
      final tempJson = await rootBundle.loadString('assets/models/temperature.json');
      final tempData = json.decode(tempJson);
      _temperature = (tempData['T'] as num).toDouble();
      print('✅ Loaded temperature: $_temperature');
    } catch (e) {
      print('⚠️ Could not load temperature.json, using T=1.0: $e');
      _temperature = 1.0;
    }
  }

  Prediction predict(Uint8List imageBytes) {
    final resized = ImagePreprocess.decodeCropResize(imageBytes);
    var probs = _runProbsFromResized(resized);

    // Apply temperature scaling for calibrated probabilities
    probs = _applyTemperature(probs);

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
        note: "Not confident. Retake a close-up leaf photo with good light.",
      );
    }

    return Prediction(
      label: _labels[best],
      confidence: bestScore,
      index: best,
      probs: probs,
    );
  }

  /// Apply temperature scaling to raw softmax probabilities
  List<double> _applyTemperature(List<double> probs) {
    if (_temperature == 1.0) return probs; // No scaling needed

    const eps = 1e-12;

    // Convert probs to logits: log(p)
    final logits = probs.map((p) => log(p.clamp(eps, 1.0))).toList();

    // Scale by temperature: logit / T
    final scaledLogits = logits.map((l) => l / _temperature).toList();

    // Subtract max for numerical stability
    final maxLogit = scaledLogits.reduce(max);
    final expLogits = scaledLogits.map((l) => exp(l - maxLogit)).toList();

    // Normalize to get calibrated probabilities
    final sumExp = expLogits.reduce((a, b) => a + b);
    return expLogits.map((e) => e / sumExp).toList();
  }

  /// Used by XAI occlusion
  List<double> runProbsForImage(img.Image resized224) {
    var probs = _runProbsFromResized(resized224);
    // Apply temperature scaling for consistency
    return _applyTemperature(probs);
  }

  List<double> _runProbsFromResized(img.Image resized224) {
    final inTensor = interpreter.getInputTensor(0);
    final outTensor = interpreter.getOutputTensor(0);

    final TensorType inType = inTensor.type;
    final TensorType outType = outTensor.type;

    final inParams = inTensor.params;   // scale/zeroPoint for quant
    final outParams = outTensor.params;

    final input = _makeInput4D(resized224, inType, inParams.scale, inParams.zeroPoint);
    final output = _makeOutput2D(labels.length, outType);

    interpreter.run(input, output);

    return _decodeOutput(output, outType, outParams.scale, outParams.zeroPoint);
  }

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
    if (outType == TensorType.float32) {
      return [List<double>.filled(n, 0.0)];
    }
    if (outType == TensorType.uint8 || outType == TensorType.int8) {
      return [List<int>.filled(n, 0)];
    }
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

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }
}
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'reshape_ext.dart';


class Prediction {
  final String label;
  final double confidence;
  final int index;
  final List<double> probs;

  Prediction({
    required this.label,
    required this.confidence,
    required this.index,
    required this.probs,
  });
}

class TomatoClassifier {
  static const int imgSize = 224;

  late final Interpreter _interpreter;
  late final List<String> _labels;

  Interpreter get interpreter => _interpreter;
  List<String> get labels => _labels;

  Future<void> load() async {
    _interpreter = await Interpreter.fromAsset('assets/models/tomato_mnv3small.tflite');

    final labelsJson = await rootBundle.loadString('assets/models/class_names.json');
    final decoded = json.decode(labelsJson) as List<dynamic>;
    _labels = decoded.map((e) => e.toString()).toList();
  }

  /// ✅ Phone-safe preprocessing:
  /// - decode
  /// - fix EXIF orientation
  /// - center-crop square
  /// - resize 224x224
  /// - float32 normalize [0,1]
  Float32List preprocess(Uint8List imageBytes) {
    img.Image? image = img.decodeImage(imageBytes);
    if (image == null) throw Exception("Could not decode image");
    image = img.bakeOrientation(image);

    final w = image.width;
    final h = image.height;
    final size = w < h ? w : h;
    final x0 = (w - size) ~/ 2;
    final y0 = (h - size) ~/ 2;

    final cropped = img.copyCrop(image, x: x0, y: y0, width: size, height: size);
    final resized = img.copyResize(cropped, width: imgSize, height: imgSize);

    final Float32List input = Float32List(imgSize * imgSize * 3);
    int i = 0;
    for (int y = 0; y < imgSize; y++) {
      for (int x = 0; x < imgSize; x++) {
        final p = resized.getPixel(x, y);
        input[i++] = p.r / 255.0;
        input[i++] = p.g / 255.0;
        input[i++] = p.b / 255.0;
      }
    }
    return input;
  }

  Prediction predict(Uint8List imageBytes) {
    final inputFlat = preprocess(imageBytes);
    final input = inputFlat.reshape([1, imgSize, imgSize, 3]);

    final output = List.filled(_labels.length, 0.0).reshape([1, _labels.length]);
    _interpreter.run(input, output);

    final probs = (output[0] as List<double>);
    int best = 0;
    double bestScore = probs[0];

    for (int i = 1; i < probs.length; i++) {
      if (probs[i] > bestScore) {
        bestScore = probs[i];
        best = i;
      }
    }

    return Prediction(
      label: _labels[best],
      confidence: bestScore,
      index: best,
      probs: probs,
    );
  }

  void close() => _interpreter.close();
}

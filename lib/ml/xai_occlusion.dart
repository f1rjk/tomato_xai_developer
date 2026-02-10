import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'reshape_ext.dart';


class OcclusionXAI {
  static const int imgSize = 224;

  /// Returns heatmap Float32List length 224*224 (values 0..1)
  static Future<Float32List> explain({
    required Uint8List originalImageBytes,
    required Interpreter interpreter,
    required int numClasses,
    required int targetClass,
    int grid = 8,
  }) async {
    img.Image? image = img.decodeImage(originalImageBytes);
    if (image == null) throw Exception("Decode failed");
    image = img.bakeOrientation(image);

    // Same crop/resize as classifier so overlay matches
    final w = image.width, h = image.height;
    final size = min(w, h);
    final x0 = (w - size) ~/ 2;
    final y0 = (h - size) ~/ 2;

    final cropped = img.copyCrop(image, x: x0, y: y0, width: size, height: size);
    final baseImg = img.copyResize(cropped, width: imgSize, height: imgSize);

    final baseline = _run(interpreter, baseImg, numClasses)[targetClass];

    final patch = (imgSize / grid).floor();
    final scores = List<double>.filled(grid * grid, 0.0);

    int idx = 0;
    for (int gy = 0; gy < grid; gy++) {
      for (int gx = 0; gx < grid; gx++) {
        final occluded = img.Image.from(baseImg);

        final xStart = gx * patch;
        final yStart = gy * patch;
        final xEnd = min(xStart + patch, imgSize);
        final yEnd = min(yStart + patch, imgSize);

        for (int y = yStart; y < yEnd; y++) {
          for (int x = xStart; x < xEnd; x++) {
            occluded.setPixelRgba(x, y, 128, 128, 128, 255);
          }
        }

        final score = _run(interpreter, occluded, numClasses)[targetClass];
        scores[idx++] = (baseline - score); // drop = importance
      }
    }

    double minV = scores.reduce(min);
    double maxV = scores.reduce(max);
    final denom = (maxV - minV).abs() < 1e-9 ? 1.0 : (maxV - minV);

    final heatmap = Float32List(imgSize * imgSize);
    for (int gy = 0; gy < grid; gy++) {
      for (int gx = 0; gx < grid; gx++) {
        final v = ((scores[gy * grid + gx] - minV) / denom).clamp(0.0, 1.0);
        final xStart = gx * patch;
        final yStart = gy * patch;
        final xEnd = min(xStart + patch, imgSize);
        final yEnd = min(yStart + patch, imgSize);

        for (int y = yStart; y < yEnd; y++) {
          for (int x = xStart; x < xEnd; x++) {
            heatmap[y * imgSize + x] = v.toDouble();
          }
        }
      }
    }

    return heatmap;
  }

  static List<double> _run(Interpreter interpreter, img.Image image, int numClasses) {
    final input = Float32List(imgSize * imgSize * 3);
    int i = 0;

    for (int y = 0; y < imgSize; y++) {
      for (int x = 0; x < imgSize; x++) {
        final p = image.getPixel(x, y);
        input[i++] = p.r / 255.0;
        input[i++] = p.g / 255.0;
        input[i++] = p.b / 255.0;
      }
    }

    final inputTensor = input.reshape([1, imgSize, imgSize, 3]);
    final output = List.filled(numClasses, 0.0).reshape([1, numClasses]);

    interpreter.run(inputTensor, output);
    return (output[0] as List<double>);
  }
}

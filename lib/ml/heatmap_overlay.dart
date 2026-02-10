import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

Uint8List overlayHeatmapV2({
  required Uint8List originalBytes,
  required Float32List heatmap, // 224*224, 0..1
  double alpha = 0.45,
  double clipLow = 0.55, // suppress weak noise
}) {
  img.Image? image = img.decodeImage(originalBytes);
  if (image == null) throw Exception("Decode failed");
  image = img.bakeOrientation(image);

  final w = image.width, h = image.height;
  final s = min(w, h);
  final x0 = (w - s) ~/ 2;
  final y0 = (h - s) ~/ 2;

  final cropped = img.copyCrop(image, x: x0, y: y0, width: s, height: s);
  final base = img.copyResize(cropped, width: 224, height: 224);

  // smooth heatmap 3x3
  final smooth = Float32List(224 * 224);
  for (int y = 0; y < 224; y++) {
    for (int x = 0; x < 224; x++) {
      double sum = 0;
      int cnt = 0;
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          final yy = y + dy, xx = x + dx;
          if (yy >= 0 && yy < 224 && xx >= 0 && xx < 224) {
            sum += heatmap[yy * 224 + xx];
            cnt++;
          }
        }
      }
      smooth[y * 224 + x] = (sum / cnt).toDouble();
    }
  }

  // clip low activations
  for (int i = 0; i < smooth.length; i++) {
    final v = smooth[i];
    smooth[i] = (v < clipLow) ? 0.0 : ((v - clipLow) / (1.0 - clipLow));
  }

  // overlay
  for (int y = 0; y < 224; y++) {
    for (int x = 0; x < 224; x++) {
      final v = smooth[y * 224 + x].clamp(0.0, 1.0);

      final r = (255 * v).toInt();
      final g = (80 * (1 - (v - 0.5).abs() * 2)).clamp(0, 80).toInt();
      final b = (255 * (1 - v)).toInt();

      final p = base.getPixel(x, y);
      final nr = (p.r * (1 - alpha) + r * alpha).toInt();
      final ng = (p.g * (1 - alpha) + g * alpha).toInt();
      final nb = (p.b * (1 - alpha) + b * alpha).toInt();
      base.setPixelRgba(x, y, nr, ng, nb, 255);
    }
  }

  return Uint8List.fromList(img.encodeJpg(base, quality: 90));
}

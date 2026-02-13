import 'dart:typed_data';
import 'package:image/image.dart' as img;

class ImagePreprocess {
  static const int size = 224;

  static img.Image decodeCropResize(Uint8List bytes) {
    img.Image? image = img.decodeImage(bytes);
    if (image == null) throw Exception("Could not decode image");
    image = img.bakeOrientation(image);

    final w = image.width;
    final h = image.height;
    final s = w < h ? w : h;

    final x0 = (w - s) ~/ 2;
    final y0 = (h - s) ~/ 2;

    final cropped = img.copyCrop(image, x: x0, y: y0, width: s, height: s);
    return img.copyResize(cropped, width: size, height: size);
  }
}

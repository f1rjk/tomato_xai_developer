import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:media_store_plus/media_store_plus.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

class RunLogger {
  static const _fileName = "runs.csv";

  static Future<File> _file() async {
    final dir = await getApplicationDocumentsDirectory();
    return File("${dir.path}/$_fileName");
  }

  /// Internal app-private path (not visible in normal file manager)
  static Future<String> path() async => (await _file()).path;

  static Future<void> append(Map<String, dynamic> row) async {
    final f = await _file();
    final exists = await f.exists();

    final cols = [
      "ts",
      "device",
      "variant",
      "tau",
      "gate_tau",
      "margin",
      "grid",
      "alpha",
      "pred_label",
      "pred_conf",
      "abstained",
      "infer_ms",
      "xai_ms",
      "image_note",
    ];

    if (!exists) {
      await f.writeAsString(cols.join(",") + "\n");
    }

    String esc(dynamic v) {
      final s = (v ?? "").toString().replaceAll('"', '""');
      return '"$s"';
    }

    final line = cols.map((c) => esc(row[c])).join(",") + "\n";
    await f.writeAsString(line, mode: FileMode.append);
  }

  /// Ensure file exists (creates header if missing)
  static Future<void> ensureCreated() async {
    final f = await _file();
    if (await f.exists()) return;

    final cols = [
      "ts",
      "device",
      "variant",
      "tau",
      "gate_tau",
      "margin",
      "grid",
      "alpha",
      "pred_label",
      "pred_conf",
      "abstained",
      "infer_ms",
      "xai_ms",
      "image_note",
    ];
    await f.writeAsString(cols.join(",") + "\n");
  }

  /// Export runs.csv to Downloads/TomatoXAI (Android 10+ scoped storage safe)
  /// Returns a human-readable message you can show in SnackBar.
  static Future<String> exportToDownloads() async {
    await ensureCreated();
    final f = await _file();

    if (!await f.exists()) {
      return "runs.csv not found (no logs yet). Do at least 1 prediction first.";
    }

    final bytes = await f.readAsBytes();
    if (bytes.isEmpty) {
      return "runs.csv is empty. Do at least 1 prediction first.";
    }

    final ts = DateTime.now()
        .toIso8601String()
        .replaceAll(":", "-")
        .replaceAll(".", "-");
    final outName = "runs_$ts.csv";

    try {
      await MediaStore.ensureInitialized();
      MediaStore.appFolder = "TomatoXAI"; // not required, but fine

      // Write a temp file first (MediaStore needs a temp path)
      final tempPath = await _writeTemp(bytes, outName);

      final mediaStore = MediaStore();
      final saved = await mediaStore.saveFile(
        tempFilePath: tempPath,
        dirType: DirType.download,
        dirName: DirName.download,
        relativePath: "TomatoXAI",
      );

      // saved may be null on failure
      if (saved == null) {
        return "❌ Export failed (MediaStore returned null). Try Share instead.";
      }

      return "✅ Exported to Downloads/TomatoXAI as $outName";
    } catch (e) {
      // fallback: share sheet
      try {
        final xfile = XFile(f.path, mimeType: "text/csv", name: outName);
        await Share.shareXFiles([xfile], text: "Tomato XAI runs.csv export");
        return "⚠️ Could not save to Downloads.\nOpened Share Sheet instead.";
      } catch (e2) {
        return "❌ Export failed: $e\nAlso share failed: $e2";
      }
    }
  }

  /// Copy the internal file path to clipboard
  static Future<String> copyPathToClipboard() async {
    final p = await path();
    await Clipboard.setData(ClipboardData(text: p));
    return p;
  }

  static Future<String> _writeTemp(Uint8List bytes, String name) async {
    final tmpDir = await getTemporaryDirectory();
    final tmpFile = File("${tmpDir.path}/$name");
    await tmpFile.writeAsBytes(bytes, flush: true);
    return tmpFile.path;
  }

  /// Optional: preview first lines for debugging
  static Future<String> peek({int maxLines = 10}) async {
    final f = await _file();
    if (!await f.exists()) return "(runs.csv not found)";
    final text = await f.readAsString();
    final lines = const LineSplitter().convert(text);
    return lines.take(maxLines).join("\n");
  }
}

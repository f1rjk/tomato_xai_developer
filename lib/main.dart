import 'package:flutter/material.dart';
import 'screens/mvp_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const TomatoXaiApp());
}

class TomatoXaiApp extends StatelessWidget {
  const TomatoXaiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: "Tomato XAI",
      theme: ThemeData(useMaterial3: true),
      home: const MvpScreen(),
    );
  }
}

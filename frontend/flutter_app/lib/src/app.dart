import 'package:flutter/material.dart';

import 'screens/home_page.dart';

class ChoreoApp extends StatelessWidget {
  const ChoreoApp({super.key});

  @override
  Widget build(BuildContext context) {
    final colorScheme = ColorScheme.fromSeed(
      seedColor: const Color(0xFFE7B8A8),
      brightness: Brightness.light,
      surface: const Color(0xFFFFFBF8),
    );

    return MaterialApp(
      title: 'Choreo App',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: colorScheme,
        scaffoldBackgroundColor: const Color(0xFFF8F2EC),
        cardTheme: const CardThemeData(
          elevation: 0,
          margin: EdgeInsets.zero,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(24)),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white.withOpacity(0.82),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(18),
            borderSide: const BorderSide(color: Color(0xFFD8C3B8)),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(18),
            borderSide: const BorderSide(color: Color(0xFFD8C3B8)),
          ),
        ),
      ),
      home: const HomePage(),
    );
  }
}

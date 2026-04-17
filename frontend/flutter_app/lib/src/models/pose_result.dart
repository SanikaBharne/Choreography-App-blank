class PoseResult {
  const PoseResult({
    required this.overview,
    required this.rows,
    required this.frames,
  });

  final Map<String, dynamic> overview;
  final List<Map<String, dynamic>> rows;
  final List<Map<String, dynamic>> frames;

  factory PoseResult.fromJson(Map<String, dynamic> json) {
    return PoseResult(
      overview: Map<String, dynamic>.from(json['overview'] as Map),
      rows: (json['rows'] as List<dynamic>).map((item) => Map<String, dynamic>.from(item as Map)).toList(),
      frames: (json['frames'] as List<dynamic>).map((item) => Map<String, dynamic>.from(item as Map)).toList(),
    );
  }
}

class AnalysisSummary {
  const AnalysisSummary({
    required this.audioUrl,
    required this.mediaType,
    required this.duration,
    required this.tempoEstimate,
    required this.beatMethod,
    required this.beats,
    required this.timelineRows,
    required this.eightCountSummaries,
  });

  final String audioUrl;
  final String mediaType;
  final double duration;
  final double? tempoEstimate;
  final String beatMethod;
  final List<double> beats;
  final List<Map<String, dynamic>> timelineRows;
  final List<Map<String, dynamic>> eightCountSummaries;

  factory AnalysisSummary.fromJson(Map<String, dynamic> json) {
    return AnalysisSummary(
      audioUrl: json['audio_url'] as String,
      mediaType: json['media_type'] as String,
      duration: (json['duration'] as num).toDouble(),
      tempoEstimate: (json['tempo_estimate'] as num?)?.toDouble(),
      beatMethod: json['beat_method'] as String,
      beats: (json['beats'] as List<dynamic>).map((item) => (item as num).toDouble()).toList(),
      timelineRows: (json['timeline_rows'] as List<dynamic>).cast<Map<String, dynamic>>(),
      eightCountSummaries: (json['eight_count_summaries'] as List<dynamic>).cast<Map<String, dynamic>>(),
    );
  }
}

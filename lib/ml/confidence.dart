class ConfidencePolicy {
  final double tau;     // best prob must be >= tau
  final double margin;  // best - secondBest must be >= margin

  const ConfidencePolicy({
    this.tau = 0.85,
    this.margin = 0.00,
  });

  bool shouldAbstain(List<double> probs) {
    if (probs.isEmpty) return true;

    double best = -1, second = -1;
    for (final p in probs) {
      if (p > best) {
        second = best;
        best = p;
      } else if (p > second) {
        second = p;
      }
    }

    if (best < tau) return true;
    if ((best - second) < margin) return true;
    return false;
  }
}

def auto_min_cluster_size(n: int) -> int:
    return max(5, min(200, int(round(0.057469 * (n ** 0.805003)))))
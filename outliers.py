import numpy
import matplotlib.pyplot as plt


def outliers_iqr(X):
    """Find outliers in univariate dataset using interquantile criterium"""
    q25 = numpy.quantile(X, 0.25)
    q75 = numpy.quantile(X, 0.75)

    # Interquantile range
    iqr = q75 - q25

    outliers = numpy.zeros(X.shape, dtype=numpy.bool)
    outliers[:] = numpy.logical_or(X[:] < (q25 - 1.5 * iqr), X[:] > (q75 + 1.5 * iqr))

    return outliers


def outliers_2sigma(X):
    """Find outliers in univariate dataset using 2sigma criterium"""
    sigma = numpy.sqrt(numpy.var(X))
    mean = numpy.mean(X)

    outliers = numpy.zeros(X.shape, dtype=numpy.bool)
    outliers[:] = numpy.abs(X[:] - mean) > 2.0 * sigma

    return outliers


def outliers_MAD(X):
    """Find outliers in univariate dataset using MAD criterium"""
    median = numpy.median(X)

    outliers = numpy.zeros(X.shape, dtype=numpy.bool)
    # Find Median Absolute Deviation
    MAD = numpy.median(numpy.abs(X - median))
    # 1.4826 * MAD = sigma for normal distribution, (2.0 * sigma) mimics the 2sigma detection
    outliers[:] = numpy.abs(X - median) > 1.4826 * 2.0 * MAD

    return outliers


def compare(outlier_detections, runs=20):

    # Prepare results dictionary
    results = {}
    for f in outlier_detections:
        # False negatives is fraction of outliers which were not identified
        # False positives is fraction of non-outliers which were identified as outliers
        results[f] = {"false_positives": 0, "false_negatives": 0}

    for i in range(runs):
        # Generate random dataset
        N = 50
        X = numpy.random.normal(0.0, 0.5, size=N)

        # Generate outliers
        N_outliers = 5
        R = 6.0
        X_out = numpy.random.normal(-R, 0.5, size=N_outliers)
        X_out = numpy.append(X_out, numpy.random.normal(R, 0.5, size=N_outliers))

        # Join dataset with outliers
        X_all = numpy.hstack((X, X_out))

        for f in outlier_detections:
            # Mark outliers using outlier detection method
            outliers = f(X_all)

            # Iterate over all outliers
            for val in outliers[-len(X_out):]:
                if val == False:
                    results[f]["false_negatives"] += 1.0 / (runs * len(X_out))

            # Clear all outliers, only false positive outliers within true dataset will remain
            outliers[-len(X_out):] = False

            results[f]["false_positives"] += (outliers == True).sum() / (runs * len(X))

    return results


print(compare([outliers_iqr, outliers_2sigma, outliers_MAD]))

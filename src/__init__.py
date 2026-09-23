"""Network Anomaly Detection System - source package.

Modules
-------
generate_data : synthetic network-flow generator (benign + attack traffic)
preprocessor  : dataset loading, cleaning, log-transform and scaling
train         : Isolation Forest / DBSCAN / LOF detectors + persistence
evaluator     : unsupervised and supervised metrics, severity, report plots
"""

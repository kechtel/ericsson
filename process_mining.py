import os

from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.algo.filtering.log.start_activities import start_activities_filter
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.traces.log import case_statistics
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.visualization.petrinet import visualizer as pn_visualizer


def parse_filename(filename):
    _, company, tweets = filename.split('.')[0].split('-')
    return company, tweets


def filter_classified_start_activities(log, company):
    tracefilter_log_neg = attributes_filter.apply_events(log, [company],
                                                         parameters={
                                                             attributes_filter.Parameters.ATTRIBUTE_KEY: "org:resource",
                                                             attributes_filter.Parameters.POSITIVE: False})
    topics = start_activities_filter.get_start_activities(tracefilter_log_neg).keys()
    return start_activities_filter.apply(log, topics)


def print_variants_count(log):
    variants_count = case_statistics.get_variant_statistics(log)
    print(sorted(variants_count, key=lambda x: x['count'], reverse=True))


def visualize_alpha_miner(log):
    net, initial_marking, final_marking = alpha_miner.apply(log)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, variant=pn_visualizer.Variants.FREQUENCY, log=log)
    pn_visualizer.view(gviz)


def visualize_heuristics_miner(log):
    net, initial_marking, final_marking = heuristics_miner.apply(log)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, variant=pn_visualizer.Variants.FREQUENCY, log=log)
    pn_visualizer.view(gviz)


def visualize_inductive_miner(log):
    net, initial_marking, final_marking = inductive_miner.apply(log)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, variant=pn_visualizer.Variants.FREQUENCY, log=log)
    pn_visualizer.view(gviz)


def visualize_directly_follows_graph(log):
    dfg = dfg_discovery.apply(log)
    gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY)
    dfg_visualization.view(gviz)


def visualize_discovery(log):
    print_variants_count(log)
    visualize_alpha_miner(log)
    visualize_heuristics_miner(log)
    visualize_inductive_miner(log)
    visualize_directly_follows_graph(log)


if __name__ == '__main__':

    for filename in os.listdir('xes'):
        log = xes_importer.apply(os.path.join('xes', filename))
        company, _ = parse_filename(filename)
        log = filter_classified_start_activities(log, company)
        log = variants_filter.apply_auto_filter(log, parameters={
            start_activities_filter.Parameters.DECREASING_FACTOR: 0.7})
        visualize_discovery(log)

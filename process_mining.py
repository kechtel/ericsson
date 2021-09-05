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


def parse_variants_filter_arg(variants_filter_name):
    if variants_filter_name.startswith('variants_percentage'):
        percentage = variants_filter_name.split('_')[-1]
        return lambda x: variants_filter.filter_log_variants_percentage(x, percentage=float(percentage))
    elif variants_filter_name.startswith('variants_auto'):
        percentage = variants_filter_name.split('_')[-1]
        return lambda x: variants_filter.apply_auto_filter(
            x, parameters={attributes_filter.Parameters.DECREASING_FACTOR: float(percentage)})
    elif variants_filter_name.startswith('variants_top'):
        k = variants_filter_name.split('_')[-1]
        return lambda x: variants_filter.filter_variants_top_k(x, int(k))


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


def apply_alpha_miner(log, path, filename):
    net, initial_marking, final_marking = alpha_miner.apply(log)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, variant=pn_visualizer.Variants.FREQUENCY, log=log)
    pn_visualizer.save(gviz, os.path.join(path, filename.format(algorithm='alpha_miner')))


def apply_heuristics_miner(log, path, filename):
    net, initial_marking, final_marking = heuristics_miner.apply(log)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, variant=pn_visualizer.Variants.FREQUENCY, log=log)
    pn_visualizer.save(gviz, os.path.join(path, filename.format(algorithm='heuristics_miner')))


def apply_inductive_miner(log, path, filename):
    net, initial_marking, final_marking = inductive_miner.apply(log)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, variant=pn_visualizer.Variants.FREQUENCY, log=log)
    pn_visualizer.save(gviz, os.path.join(path, filename.format(algorithm='inductive_miner')))


def apply_inductive_miner_imf(log, path, filename):
    net, initial_marking, final_marking = inductive_miner.apply(log, variant=inductive_miner.Variants.IMf)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, variant=pn_visualizer.Variants.FREQUENCY, log=log)
    pn_visualizer.save(gviz, os.path.join(path, filename.format(algorithm='inductive_miner_infrequent')))


def apply_inductive_miner_imd(log, path, filename):
    net, initial_marking, final_marking = inductive_miner.apply(log, variant=inductive_miner.Variants.IMd)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, variant=pn_visualizer.Variants.FREQUENCY, log=log)
    pn_visualizer.save(gviz, os.path.join(path, filename.format(algorithm='inductive_miner_dfg')))


def apply_directly_follows_graph(log, path, filename):
    dfg = dfg_discovery.apply(log)
    gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY)
    dfg_visualization.save(gviz, os.path.join(path, filename.format(algorithm='directly_follows_graph')))


def process_discovery(log, path, filename):
    print_variants_count(log)
    apply_alpha_miner(log, path, filename)
    apply_heuristics_miner(log, path, filename)
    apply_inductive_miner(log, path, filename)
    apply_inductive_miner_imf(log, path, filename)
    apply_inductive_miner_imd(log, path, filename)
    apply_directly_follows_graph(log, path, filename)


if __name__ == '__main__':

    filters = ['variants_percentage_1.0', 'variants_top_5', 'variants_top_6', 'variants_top_7', 'variants_top_8',
               'variants_top_9', 'variants_top_10', 'variants_top_15']

    if not os.path.exists(os.path.join('results', 'process-discovery')):
        os.mkdir(os.path.join('results', 'process-discovery'))

    for filename in [filename for filename in os.listdir('xes') if filename.startswith('twcs')]:
        log = xes_importer.apply(os.path.join('xes', filename))
        company, _ = parse_filename(filename)
        log = filter_classified_start_activities(log, company)
        for variants_filter_name in filters:
            filter = parse_variants_filter_arg(variants_filter_name)
            filtered_log = filter(log)
            process_discovery_filename = \
                '{company}-{variants_filter}-{{algorithm}}.png'.format(company=company,
                                                                       variants_filter=variants_filter_name)
            process_discovery(filtered_log, os.path.join('results', 'process-discovery'), process_discovery_filename)

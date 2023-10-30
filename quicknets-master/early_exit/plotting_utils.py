import torch
import plotly.graph_objects as go
from eval_utils import get_results


def get_learning_curve_graph(model_name, exits, train_method):
    learning_curves = []
    lc_lenths = []
    testing_curves = []
    if train_method == 'backbone_first':
        learning_curves = torch.load('../results/' + model_name + '_learning_curve')
        testing_curves = torch.load('../results/' + model_name + '_testing_curve')
        learning_curves = [float(i) for i in learning_curves]
        testing_curves = [float(i) for i in testing_curves]
    elif train_method == 'layer_wise':
        for i in range(exits):
            learning_curve = torch.load('../results/' + model_name + '_exit' + str(i+1) + '_learning_curve')
            learning_curves += learning_curve
            lc_lenths.append(len(learning_curve))
            learning_curve = torch.load('../results/' + model_name + '_exit' + str(i + 1) + '_testing_curve')
            testing_curves += learning_curve
        learning_curves = [float(i) for i in learning_curves]
        testing_curves = [float(i) for i in testing_curves]

    elif train_method == 'growing':
        i = 0
        next_i = 2
        while i < exits:
            learning_curve = torch.load('../results/' + model_name + '_exit' + str(i+1) + '_learning_curve')
            learning_curves += learning_curve
            lc_lenths.append(len(learning_curve))
            learning_curve = torch.load('../results/' + model_name + '_exit' + str(i + 1) + '_testing_curve')
            testing_curves += learning_curve
            i = next_i + i
            next_i += 1
        learning_curves = [float(i) for i in learning_curves]
        testing_curves = [float(i) for i in testing_curves]


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(learning_curves))), y=learning_curves,  mode='lines', name='training'))
    fig.add_trace(go.Scatter(x=list(range(len(testing_curves))), y=testing_curves,  mode='lines', name='testing'))
    current = 0
    for i, l in enumerate(lc_lenths):
        current += l
        fig.add_vline(x=current, annotation_text="Layer" + str(i+1) + " trained", line_dash="dot")
    fig.update_layout(title='Accuracy vs. Epochs', xaxis_title='Epochs', yaxis_title='Accuracy')
    return fig


def get_figures(train_loader, test_loader, model_name, num_exits, threshold, nets, method='decision', train_method='layer_wise', break_ties = 'max_confidence'):
    exit_train_accuracy, train_accuracy, exits, not_classified_exits = get_results(train_loader, threshold, method, nets, break_ties)
    exit_test_accuracy, test_accuracy, test_exits, test_not_classified_exits = get_results(test_loader, threshold, method, nets, break_ties)
    fig1 = get_learning_curve_graph(model_name, num_exits, train_method)
    fig1.add_hline(y=train_accuracy * 100, annotation_text="Combined train accuracy", line_dash="dot")
    fig1.add_hline(y=test_accuracy * 100, annotation_text="Combined test accuracy", line_dash="dot")

    if train_method == 'layer_wise' or train_method == 'growing':
        samples = torch.load('../results/' + model_name + '_samples_trained')
        samples = samples[:-1]
        x = ['Layer' + str(i+1) for i in range(len(samples))]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=x, y=samples, name='Samples trained'))
        fig2.add_trace(go.Bar(x=x, y=exits, name='Samples early exited'))
        fig2.add_trace(
            go.Bar(x=x, y=not_classified_exits, name='Sampels exited at the end'))
        fig2.update_layout(title='Number of samples per exit (training)',
                           xaxis_title='Layer / exit',
                           yaxis_title='Number of samples')
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=x, y=test_exits, name='Samples early exited'))
        fig3.add_trace(
            go.Bar(x=x, y=test_not_classified_exits, name='Sampels exited at the end'))
        fig3.update_layout(title='Number of samples per exit (testing)',
                           xaxis_title='Layer / exit',
                           yaxis_title='Number of samples')

        return fig1, fig2, fig3, train_accuracy, test_accuracy, exit_train_accuracy, exit_test_accuracy, exits, not_classified_exits, test_exits, test_not_classified_exits
    else:
        x = ['Layer' + str(i + 1) for i in range(len(nets))]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=x, y=exits, name='Samples early exited'))
        fig2.add_trace(
            go.Bar(x=x, y=not_classified_exits, name='Samples exited at the end'))
        fig2.update_layout(title='Number of samples per exit (training)',
                           xaxis_title='Layer / exit',
                           yaxis_title='Number of samples')
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=x, y=test_exits, name='Samples early exited'))
        fig3.add_trace(
            go.Bar(x=x, y=test_not_classified_exits, name='Sampels exited at the end'))
        fig3.update_layout(title='Number of samples per exit (testing)',
                           xaxis_title='Layer / exit',
                           yaxis_title='Number of samples')

        return fig1,fig2, fig3, train_accuracy, test_accuracy, exit_train_accuracy, exit_test_accuracy, exits, not_classified_exits, test_exits, test_not_classified_exits

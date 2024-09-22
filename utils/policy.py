def greedy_action(Q_table, current_state):
    best_action = None
    max_q_value = float('-inf')
    
    # # Case 1: Q_table is a dictionary
    # if isinstance(Q_table, dict):
    #     for (state, action), q_value in Q_table.items():
    #         if state == current_state and q_value > max_q_value:
    #             max_q_value = q_value
    #             best_action = action
                
    # # Case 2: Q_table is a list of tuples
    # elif isinstance(Q_table, list):
    #     for item in Q_table:
    #         if isinstance(item, tuple) and len(item) == 2:
    #             (state, action), q_value = item
    #             if state == current_state and q_value > max_q_value:
    #                 max_q_value = q_value
    #                 best_action = action
                    
    # # Case 3: Q_table is a list of (state, Q_value) pairs
    # elif isinstance(Q_table, list):
    #     for (state, action, q_value) in Q_table:
    #         if state == current_state and q_value > max_q_value:
    #             max_q_value = q_value
    #             best_action = action
    
    return best_action
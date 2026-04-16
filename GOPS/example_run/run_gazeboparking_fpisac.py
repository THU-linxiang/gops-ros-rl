from gops.sys_simulator.sys_run import PolicyRunner

runner = PolicyRunner(
    log_policy_dir_list=["/home/lin/combined-repo/GOPS/results/pyth_gazebo_parking/DSACT_260414-105210"] ,  # directory of trained policy
    trained_policy_iteration_list=["60000"],  # iteration number of trained policy
    is_init_info=False,  # customize initial information
    #init_info={"init_state":[0.5, 3.9, 0.2,0,0,   0.5,1.8,  -0.2,2.7,  1.4,3.1,  0,0.5,1.57]},  # initial state
    legend_list=["60000"],
    save_render=False,
    constrained_env=True,

)

runner.run()

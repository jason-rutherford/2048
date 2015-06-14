// Wait till the browser is ready to render the game (avoids glitches)
window.requestAnimationFrame(function () {
  window.game = new GameManager(4, KeyboardInputManager, HTMLActuator, LocalStorageManager);

  // Credit
  // http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html

  // 1 input for each tile, square board
  var num_inputs = Math.pow(game.grid.size, 2);

  // 4 directions agent can shift tiles
  var num_actions = 4;

  // amount of temporal memory. 0 = agent lives in-the-moment :)
  var temporal_window = 1;

  var network_size = num_inputs * temporal_window +
    num_actions * temporal_window + num_inputs;

  // the value function network computes a value of taking any of the possible actions
  // given an input state. Here we specify one explicitly the hard way
  // but user could also equivalently instead use opt.hidden_layer_sizes = [20,20]
  // to just insert simple relu hidden layers.
  var layer_defs = [];

  layer_defs.push({
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: network_size
  });
  layer_defs.push({type: 'fc', num_neurons: 50, activation: 'relu'});
  layer_defs.push({type: 'fc', num_neurons: 50, activation: 'relu'});
  layer_defs.push({type: 'regression', num_neurons: num_actions});

  // options for the Temporal Difference learner that trains the above net
  // by backprogagating the temporal difference learning rule.
  var tdtrainer_options = {
    learning_rate: 0.001,
    momentum: 0.0,
    batch_size: 64,
    l2_decay: 0.01
  };

  // agent parameter spec to play with
  var opt = {
    alpha: 0.01,                          // value function learning rate
    update: 'qlearn',                     // qlearn | sarsa
    epsilon: 0.2,                         // initial epsilon for epsilon-greedy policy, [0, 1)
    epsilon_min: 0.05,                    // probability for random actions
    epsilon_test_time: 0.05,              // don't make any random choices, ever
    experience_add_every: 5,              // number of time steps before we add another experience to replay memory
    experience_size: 10000,               // size of experience replay memory
    gamma: 0.9,                           // discount factor, [0, 1)
    layer_defs: layer_defs,
    learning_steps_per_iteration: 20,
    learning_steps_total: 200000,
    learning_steps_burnin: 3000,
    num_hidden_units: 10,                 // number of neurons in hidden layer
    start_learn_threshold: 1000,
    tderror_clamp: 1.0,                   // for robustness
    temporal_window: temporal_window,
    tdtrainer_options: tdtrainer_options
  };

  var brain = new deepqlearn.Brain(num_inputs, num_actions, opt); // woohoo

// todo: this should resemble the env object from the docs, not done
   var World = function() {
    this.reset();
  };
  World.prototype = {
    reset: function() {
      game.restart();
    },
    getNumStates: function() {
      return Math.pow(game.grid.size, 2); // x,y,vx,vy, puck dx,dy
    },
    getMaxNumActions: function() {
      return 4; // left, right, up, down
    }
  };

  // do actions
  setInterval(function() {
    var action = brain.forward();

    // See KeyboardInputManager keydown event listener
    // 0 up, 1 right, 2 down, 3 left
    game.inputManager.emit('move', action);
    
    // todo: compute a reward, use score and edge calc
    // sum of difference between tile edges 
    // brain.backward();  // see docs
  }, 200);
});

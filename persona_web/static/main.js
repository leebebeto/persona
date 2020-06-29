let {Router, Route, Link, browserHistory} = window.ReactRouter;



function clicked(e){
  fetch('', {  
    method: 'POST',
    body: JSON.stringify({
      a: '123',
      b: '456',
    })
  })
}

function appChange(e){
  fetch('', {  
    method: 'POST',
    body: JSON.stringify({
      a: '123',
      b: '456',
    })
  })
}

class Index extends React.Component {
  constructor(props) {
        super(props);
        this.state = {
          temp : document.getElementsByTagName('script')[4].getAttribute('text'),
          testVariable : {value: "this is a test"},
          data : {    
          name: 'John Smith',
          imgURL: 'http://lorempixel.com/100/100/',
          hobbyList: ['coding', 'writing', 'skiing']
          }
      }

        console.log(this.state.temp);
        console.log(this.state.testVariable.value);
        console.log(this.state.data.name);

  }





  render() {
    return (
      <div>
        <a onClick={clicked}>Hello React!</a>
        <h1>{this.state.testVariable.value}</h1>
        <h1>{this.state.data.name}</h1>
        <input type="text" name="id" placeholder="ID" value={'567'} onChange={appChange} />
        <input type="password" name="password" placeholder="pwd" value={'789'} onChange={appChange} />
      </div>
    );
  }
}


ReactDOM.render((
  <Router history={browserHistory}>
    <Route path='/' component={Index}>
    </Route>
  </Router>
  ), document.getElementById('content')
);

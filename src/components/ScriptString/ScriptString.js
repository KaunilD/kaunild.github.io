import React from 'react'

class ScriptString extends React.Component {
  constructor (props) {
    super(props)
    this.state = {
      initialString: props.string,
      string: '.'
    }

    this.scriptString = this.scriptString.bind(this)
  }

  componentDidMount () {
    this.scriptString(this.state.initialString, 0)
  }

  scriptString (string, index) {
    var parent = this
    if (index < string.length) {
      setTimeout(function () {
        parent.setState({string: string.substr(0, index + 1)})
        parent.scriptString(string, index + 1)
      }, 100)
    } else {
      this.props.onComplete()
    }
  }

  render () {
    return (
      <span>
        {this.state.string}
      </span>
    )
  }
}

export default ScriptString

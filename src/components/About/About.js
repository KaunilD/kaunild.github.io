import React from 'react'

import './About.css'

class About extends React.Component {
  constructor (props) {
    super(props)
    this.state = {
      page1Descp: false,
      page1Seperator: false,
      page1Overlay: false
    }
  }

  componentWillReceiveProps (nextProps) {
    this.setState(nextProps)
  }
  componentDidMount () {
  }
  render () {
    return (
      <div className='page-content'>
        <div className='page-content-inner'>

          {/*<div className={this.state.page1Seperator ? 'page-seperator show-page-seperator' : 'page-seperator'} />*/}
          <div className={this.state.page1Descp ? 'page-descp show-page-descp' : 'page-descp'}>
            <p>
            I'm a curious Software Engineer
            <br/>
            passionate about building tools and solving challenges that improve people's lives.
            </p>
            <div className='left interests'>
              <span className='title-hi'>My areas of</span>
              <span className='h1'>RESEARCH </span>
              <span className='h4'>Machine Learning</span>
              <span className='h4'>Computer Vision</span>
              <span className='h4'>Data Science</span>
            </div>
            <div className='right education'>
              <span className='title-hi'>My background</span>
              <span className='h1'>EDUCATION</span>
              <span className='h4'>B.Tech. in</span>
              <span className='h4'>CS</span>
              <span className='h4'>University of Mumbai</span>
            </div>
          </div>
        </div>
      </div>
    )
  }
}

export default About

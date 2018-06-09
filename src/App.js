import React, { Component } from 'react'
import ReactDOM from 'react-dom'
import Particles from 'react-particles-js'

import ScriptString from './components/ScriptString'
import './App.css'

var smoothScroll = require('smoothscroll')

class App extends Component {

  constructor (props) {
    super(props)
    this.state = {
      title: 'Kaunil Dhruv',
      subTitle: 'SOFTWARE ENGINEER',
      socialLinks: false,
      page1Descp: false,
      page1Seperator: false,
      page1Title: false,
      page1Overlay: false,
      page2Overlay: false,
      page3Overlay: false,
      page4Overlay: false,
      page5Overlay: false,
      profileText: false,
      menu: false
    }

    this.onScroll = this.onScroll.bind(this)
  }

  componentDidMount () {
  }

  toggleSocialLinks () {
    this.setState({socialLinks: true, profileText: true})
    setTimeout(this.initPage1Visibility(), 1000)
  }

  initPage1Visibility () {
    let parent = this
    setTimeout(function () {
      parent.setState({menu: true, page1Overlay: true, page1Title: true, page1Descp: true, page1Seperator: true})
    }, 300)
  }

  onScroll () {
    let scrollTop = ReactDOM.findDOMNode(this.content).scrollTop
    console.log(scrollTop, ReactDOM.findDOMNode(this.page1).clientHeight)
    if (scrollTop >= ReactDOM.findDOMNode(this.page1).clientHeight) {
      this.setState({page2Overlay: true})
    }
    if (scrollTop >= ReactDOM.findDOMNode(this.page1).clientHeight + ReactDOM.findDOMNode(this.page2).clientHeight) {
      this.setState({page3Overlay: true})
    }

    if (scrollTop >= ReactDOM.findDOMNode(this.page1).clientHeight + ReactDOM.findDOMNode(this.page2).clientHeight + ReactDOM.findDOMNode(this.page3).clientHeight) {
      this.setState({page4Overlay: true})
    }
    if (scrollTop >= ReactDOM.findDOMNode(this.page1).clientHeight + ReactDOM.findDOMNode(this.page2).clientHeight + ReactDOM.findDOMNode(this.page3).clientHeight + ReactDOM.findDOMNode(this.page5).clientHeight) {
      this.setState({page5Overlay: true})
    }
  }

  handleLinkClick (event, target) {
    event.preventDefault()
    smoothScroll(target, 500, null, this.content)
    console.log(event)
  }

  render () {
    return (
      <div className='content-wrapper'>
        <div className='sidebar'>
          <div className='profile'>
            <div
              className={this.state.profileText ? 'profile-text slideup-profile-text' : 'profile-text'}
              ref={(element) => this.profileText = element}
              >
              <div className='title'>
                <span className='title-hi'>Hi I'm</span>
                <ScriptString onComplete={() => this.toggleSocialLinks()} string="KAUNIL DHRUV"/>
              </div>
            </div>

            <div className={this.state.menu ? 'menu show-menu' : 'menu'}>
              <ul className='menu-main'>
                <li>
                  <a onClick={(e) => this.handleLinkClick(e, this.page1)}>
                    About
                  </a>
                </li>
                <li>
                  <a onClick={(e) => this.handleLinkClick(e, this.page2)}>Publications</a>
                </li>
                <li>
                  <a onClick={(e) => this.handleLinkClick(e, this.page3)}>Projects</a>
                  <ul className='sub-menu'>
                    <li>
                      COBRIX
                    </li>
                    <li>
                      DISGUISED FACE DETECTION
                    </li>
                    <li>
                      SMART TITLE
                    </li>
                    <li>
                      L.M.S
                    </li>
                    <li>
                      DIGITIZING MEDICAL REPORTS
                    </li>
                    <li>
                      STANDALONE MULTIMEDIA BOX
                    </li>
                    <li>
                      APTITIDE CMS
                    </li>
                  </ul>
                </li>
                <li>
                  <a onClick={(e) => this.handleLinkClick(e, this.page4)}>Experiments</a>
                  <ul className='sub-menu'>
                    <li>
                      REACT JS
                    </li>
                    <li>
                      MACHINE LEARNING
                    </li>
                    <li>
                      COMPUTER VISION
                    </li>
                  </ul>
                </li>

                <li onClick={(e) => this.handleLinkClick(e, this.page5)}>
                  <a>Resume</a>
                </li>
                <li>
                  <a onClick={(e) => this.handleLinkClick(e, this.page6)}>Contact</a>
                </li>
              </ul>
            </div>



            <div ref={(elt) => this.socialLinks = elt} className={this.state.socialLinks ? 'social-links show-social-links' : 'social-links'}>
              <ul>
                <li>
                  <img src='assets/github.png' />
                </li>
                <li>
                  <img src='assets/linkedin.png' />
                </li>
                <li>
                  <img src='assets/scholar.png' />
                </li>
              </ul>
            </div>
            <div className='notices'>
              <p>Built with <img src='react.png' /></p>
            </div>
          </div>


        </div>
        <div className='content' onScroll={this.onScroll} ref={(element) => this.content = element}>
          <div className='page-1' ref={(element) => this.page1 = element}>
            <div className={!this.state.page1Overlay ? 'page-1-overlay' : 'page-1-overlay hide-page-1-overlay'} />
            <div className='page-content'>
              <div className='page-content-inner'>
                <div className={this.state.page1Title ? 'page-title show-page-title' : 'page-title'}>
                </div>
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
          </div>

          <div className='page-2' ref={(element) => this.page2 = element}>
            <div className={!this.state.page2Overlay ? 'page-2-overlay' : 'page-2-overlay hide-page-2-overlay'} />
            <div className='page-content'>
              <div className='page-content-inner'>
                <div className={this.state.page1Title ? 'page-title show-page-title' : 'page-title'}>
                  <p data-value='PUBLICATIONS'>
                    PUBLICATIONS
                  </p>
                </div>

                {/*<div className={this.state.page1Seperator ? 'page-seperator show-page-seperator' : 'page-seperator'} />*/}

                <div className={this.state.page1Descp ? 'page-descp show-page-descp' : 'page-descp'}>
                  <div className='page-item-full'>
                    <span className='h4'>Generative Street Addresses from Satellite Imagery</span>
                    <span className='h6'>March 08, 2018</span>
                    <span className='h6 light'>
                      Ilke Demir, Forest Hughes, Aman Raj, <span className='dark'>Kaunil Dhruv</span>,<br/>
                      Suryanarayana Murthy Muddala, Sanyam Garg,<br/>
                      Barrett Doo, Ramesh Raskar
                    </span>
                  </div>
                  <div className='page-item-full'>
                    <span className='h4'>Generative Street Addresses from Satellite Imagery</span>
                    <span className='h6'>March 08, 2018</span>
                    <span className='h6 light'>
                      Ilke Demir, Forest Hughes, Aman Raj, <span className='dark'>Kaunil Dhruv</span>,<br/>
                      Suryanarayana Murthy Muddala, Sanyam Garg,<br/>
                      Barrett Doo, Ramesh Raskar
                    </span>
                  </div>
                </div>


              </div>
            </div>
          </div>

          <div className='page-3' ref={(element) => this.page3 = element}>
            <div className={!this.state.page3Overlay ? 'page-3-overlay' : 'page-3-overlay hide-page-3-overlay'} />
          </div>

          <div className='page-4' ref={(element) => this.page4 = element}>
            <div className={!this.state.page4Overlay ? 'page-4-overlay' : 'page-4-overlay hide-page-4-overlay'} />
          </div>

          <div className='page-5' ref={(element) => this.page5 = element}>
            <div className={!this.state.page5Overlay ? 'page-5-overlay' : 'page-5-overlay hide-page-5-overlay'} />
          </div>
        </div>
      </div>
    )
  }
}

export default App;

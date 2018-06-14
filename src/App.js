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
      page2Descp: false,
      page3Descp: true,
      page4Descp: true,
      page5Descp: true,
      page1Seperator: false,
      page1Title: false,
      page2Title: false,
      page3Title: true,
      page4Title: true,
      page5Title: true,
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
    smoothScroll(
      target,
      500,
      null,
      this.content
    )
  }

  handleNavItemClick (event) {
    event.preventDefault()
    if (ReactDOM.findDOMNode(event.target).classList.contains('nav_item-current')) {
      return false
    }
    let items = ReactDOM.findDOMNode(event.target).parentNode.childNodes
    items.forEach((item) => {
      item.classList.remove('nav_item-current')
    })
    ReactDOM.findDOMNode(event.target).classList.add('nav_item-current')
    smoothScroll(
      document.getElementById(event.target.getAttribute('data')).getBoundingClientRect().top + this.publications.getBoundingClientRect().top,
      500,
      null,
      ReactDOM.findDOMNode(this.publications)
    )
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
            <div className='page-content'>
              <div className='page-content-inner'>
                <div className={this.state.page1Title ? 'page-title show-page-title' : 'page-title'}>
                  <p data-value='PUBLICATIONS'>
                    PUBLICATIONS
                  </p>
                </div>

                <div className={this.state.page1Descp ? 'page-descp show-page-descp' : 'page-descp'}>
                  <nav  class='page-navigation'>
                    <ul>
                      <li onClick={(e) => this.handleNavItemClick(e)} className='nav_item nav_item-current' data='01'><span class="nav_item-title">01</span></li>
                      <li onClick={(e) => this.handleNavItemClick(e)} className='nav_item' data='02'><span class="nav_item-title">02</span></li>
                    </ul>
                  </nav>
                  <div className='page-items-container' ref={(element) => this.publications = element}>

                    <div className='page-item-full' id='01'>
                      <div>
                        <span className='h4'>Generative Street Addresses from<br />Satellite Imagery</span>
                        <span className='h6 light'>March 08, 2018</span>
                        <span className='h6'>
                          Ilke Demir, Forest Hughes, Aman Raj, <span className='underline'>Kaunil Dhruv</span>
                        </span>
                        <span className='h6'>
                          Suryanarayana Murthy, Sanyam Garg, Barrett Doo
                        </span>
                        <span className='h6'>
                          Ramesh Raskar
                        </span>
                        <br />
                        <span className='h4'>Abstract</span>
                        <span className='h6 justify'>
                          We describe our automatic generative algorithm to create street addresses (Robocodes) from satellite images by learning and labeling regions, roads, and blocks. 75% of the world lacks street addresses [12 ]. According to the United Nations, this means 4 billion people are ‘invisible’. Recent initiatives tend to name unknown areas by geocoding, which uses latitude and longitude information. Nevertheless settlements abut roads and such addressing schemes are not coherent with the road topology. Instead, our algorithm starts with extracting roads and junctions from satellite imagery utilizing deep learning. Then, it uniquely labels the regions, roads, and houses using some graph- and proximity-based algorithms. We present our results on both cities in mapped areas and in developing countries. We also compare productivity based on current ad-hoc and new complete addresses. We conclude with contrasting our generative addresses to current industrial and open solutions.
                        </span>
                    </div>
                    </div>

                    <div className='page-item-full' id='02'>
                      <div>

                        <span className='h4'>Robocodes: Towards Generative Street<br/>Addresses from Satellite Imagery</span>
                        <span className='h6 light'>CVPR 2017 - July, 2017</span>
                        <span className='h6'>
                          Ilke Demir, Forest Hughes, Aman Raj
                        </span>
                        <span className='h6'>
                          Kleovoulos Tsourides, Divyaa Ravichandran, <span className='underline'>Kaunil Dhruv</span>
                        </span>
                        <span className='h6'>
                         Suryanarayana Murthy, Sanyam Garg, Jatin Malhotra
                        </span>
                        <span className='h6'>
                          Barrett Doo, Grace Kermani, Ramesh Raskar
                        </span>
                        <span className='h4'>Abstract</span>
                        <span className='h6 justify'>
                          We describe our automatic generative algorithm to create street addresses from satellite images by learning and labeling roads, regions, and address cells. Currently, 75% of the world’s roads lack adequate street addressing systems. Recent geocoding initiatives tend to convert pure latitude and longitude information into a memorable form for unknown areas. However, settlements are identified by streets, and such addressing schemes are not coherent with the road topology. Instead, we propose a generative address design that maps the globe in accordance with streets. Our algorithm starts with extracting roads from satellite imagery by utilizing deep learning. Then, it uniquely labels the regions, roads, and structures using some graph- and proximity-based algorithms. We also extend our addressing scheme to (i) cover inaccessible areas following similar design principles; (ii) be inclusive and flexible for changes on the ground; and (iii) lead as a pioneer for a unified street-based global geodatabase. We present our results on an example of a developed city and multiple undeveloped cities. We also compare productivity on the basis of current ad hoc and new complete addresses. We conclude by contrasting our generative addresses to current industrial and open solutions.
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className='page-3' ref={(element) => this.page3 = element}>
            <div className='page-content'>
              <div className='page-content-inner'>
                <div className={this.state.page3Title ? 'page-title show-page-title' : 'page-title'}>
                  <p data-value='PROJECTS'>
                    PORJECTS
                  </p>
                </div>
                <div className={this.state.page3Descp ? 'page-descp show-page-descp' : 'page-descp'}>
                  <nav  class='page-navigation'>
                    <ul>
                      <li onClick={(e) => this.handleNavItemClick(e)} className='nav_item nav_item-current' data='01'><span class="nav_item-title">01</span></li>
                      <li onClick={(e) => this.handleNavItemClick(e)} className='nav_item' data='02'><span class="nav_item-title">02</span></li>
                      <li onClick={(e) => this.handleNavItemClick(e)} className='nav_item' data='03'><span class="nav_item-title">03</span></li>
                      <li onClick={(e) => this.handleNavItemClick(e)} className='nav_item' data='02'><span class="nav_item-title">04</span></li>
                      <li onClick={(e) => this.handleNavItemClick(e)} className='nav_item' data='01'><span class="nav_item-title">05</span></li>
                      <li onClick={(e) => this.handleNavItemClick(e)} className='nav_item' data='02'><span class="nav_item-title">05</span></li>
                    </ul>
                  </nav>
                  <div className='page-items-container' ref={(element) => this.projects = element}>

                    <div className='page-item-full' id='01'>
                      <div>
                        <span className='h4'>COBRIX</span>
                      </div>
                    </div>
                    <div className='page-item-full' id='02'>
                      <div>
                        <span className='h4'>DISGUISED FACE DETECTION</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className='page-4' ref={(element) => this.page4 = element}>
            <div className='page-content'>
              <div className='page-content-inner'>
                <div className={this.state.page4Title ? 'page-title show-page-title' : 'page-title'}>
                  <p data-value='EXPERIMENTS'>
                    EXPERIMENTS
                  </p>
                </div>
                <div className={this.state.page4Descp ? 'page-descp show-page-descp' : 'page-descp'}>
                  <nav  class='page-navigation'>
                    <ul>
                      <li onClick={(e) => this.handleNavItemClick(e)} className='nav_item nav_item-current' data='01'><span class="nav_item-title">01</span></li>
                      <li onClick={(e) => this.handleNavItemClick(e)} className='nav_item' data='02'><span class="nav_item-title">02</span></li>
                      <li onClick={(e) => this.handleNavItemClick(e)} className='nav_item' data='03'><span class="nav_item-title">03</span></li>
                    </ul>
                  </nav>
                  <div className='page-items-container' ref={(element) => this.experiments = element}>

                    <div className='page-item-full' id='01'>
                      <div>
                        <span className='h4'>MACHINE LEARNING</span>
                      </div>
                    </div>
                    <div className='page-item-full' id='02'>
                      <div>
                        <span className='h4'>REACT JS</span>
                      </div>
                    </div>
                    <div className='page-item-full' id='03'>
                      <div>
                        <span className='h4'>COMPUTER VISION</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
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

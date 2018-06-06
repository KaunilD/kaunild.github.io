import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
class App extends Component {
  render () {
    return (
      <div className='content-wrapper'>
        <div class="sidebar">
          <div className='profile'>
            <div className='profile-text'>
              <div className='title'>
                Kaunil Dhruv
              </div>
              <div className='subtitle'>
                Software Engineer
              </div>
            </div>

            <div className='social-links'>
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
          </div>
        </div>
        <div class="content">
        </div>
      </div>
    )
  }
}

export default App;

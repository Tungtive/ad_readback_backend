import { useState } from 'react';
import {
  MemoryRouter as Router,
  Routes,
  Route,
  useNavigate,
} from 'react-router-dom';
import Nav1 from './Nav1';
import Nav2 from './Nav2';
import Nav3 from './Nav3';
import Nav4 from './Nav4';
import logo from '../img/logo.svg';
import sty from './home.module.css';

export default function Home() {
  const tabList = [
    {
      label: 'Nav1',
      key: '1',
      path: '/home/nav1',
    },
    {
      label: 'Nav2',
      key: '2',
      path: '/home/nav2',
    },
    {
      label: 'Nav3',
      key: '3',
      path: '/home/nav3',
    },
    {
      label: 'Nav4',
      key: '4',
      path: '/home/nav4',
    },
  ];

  const [curTab, setCurTab] = useState('1');
  const navigate = useNavigate();
  return (
    <div className={sty.box}>
      <div className={sty.headerBox}>
        <div className={sty.headerCenter}>
          <div className={sty.headerLeft}>
            <img src={logo} className={sty.logo} alt="" srcset="" />
            <h3 className={sty.h3}>Name</h3>
            {tabList.map((v) => {
              return (
                <div
                  key={v.key}
                  className={`${sty.navItem} ${
                    curTab == v.key ? sty.navItemActive : ''
                  }`}
                  onClick={() => {
                    setCurTab(v.key);
                    navigate(v.path);
                  }}
                >
                  {v.label}
                </div>
              );
            })}
          </div>
          <div className={sty.headerRight}>
            <div className={sty.nickname}>Tom</div>
            <div onClick={() => {
                navigate("/");
            }} className={sty.out}>Logout</div>
          </div>
        </div>
      </div>

      <div>
        <Routes>
          {/* <Route path="*" element={<Nav1 />} /> */}
          <Route path="nav1" element={<Nav1 />} />
          <Route path="nav2" element={<Nav2 />} />
          <Route path="nav3" element={<Nav3 />} />
          <Route path="nav4" element={<Nav4 />} />
        </Routes>
      </div>
    </div>
  );
}
